#  #!/usr/bin/env python
#  -*- coding: utf-8 -*-
#  ******************************************************************************
#    Copyright (c) 2022.
#    Developed by Yifei Lu
#    Last change on 1/17/22, 11:21 AM
#    Last change by yifei
#   *****************************************************************************
import logging
from .node import *

class ReverseFlowError(Exception):
    """Exception raised when reverse flow is detected in the compressor."""
    pass


class Compressor:
    """
    Class to formulate compressor stations.
    """

    def __init__(
        self,
        compressor_index: int,
        inlet: Node,
        outlet: Node,
        drive="electric",
        compression_ratio=1.1,
        thermodynamic_process="isentropic",
        efficiency=0.85,
        gas_density=None,
        control_mode="ratio",
        p_out_target=None,
    ):
        self.compressor_index = compressor_index
        self.inlet = inlet
        self.outlet = outlet
        self.inlet_index = inlet.index
        self.outlet_index = outlet.index
        self.drive = drive
        self.cp = inlet.gas_mixture.cp if hasattr(inlet.gas_mixture, 'cp') else 2142  # J/kg·K
        self.cv = inlet.gas_mixture.cv if hasattr(inlet.gas_mixture, 'cv') else 1648  # J/kg·K
        self.T1 = inlet.temperature
        self.compression_ratio = compression_ratio
        self.efficiency = efficiency
        self.gas_density = gas_density or inlet.gas_mixture.density
        self.control_mode = control_mode
        self.p_out_target = p_out_target
        
        if thermodynamic_process == "isentropic":
            self.n = self.cp / self.cv
        elif thermodynamic_process == "isothermal":
            self.n = 1
        else:
            raise ValueError(
                "Only isentropic or isothermal process is currently supported."
            )

        self.mass_flow_rate = None
        self.flow_rate = None  # Volumetric flow rate
        self.power_consumption_value = 0.0
        self.gas_mixture = inlet.gas_mixture
        
        # Gas composition tracking (for compatibility with Pipeline interface)
        self.outflow_composition = inlet.gas_mixture.eos_composition if hasattr(inlet.gas_mixture, 'eos_composition') else None

    def update_flow_rate(self, total_flow_in, total_flow_out, inlet_node_demand, outlet_node_demand):
        """Update compressor flow rate based on mass balance.

        Which side determines the flow depends on which node balances are
        actually enforced.  The residual assembled in PressureProblem._compute
        covers JUNCTION nodes only — reference nodes are dropped, so their
        balance is free and carries no information.

        * Outlet is a reference node (pressure-control mode, where the plugin
          splits the station and pins the outlet at p_out_target).  The outlet
          balance is not enforced, so averaging it in would tie half of the
          inlet node's equation to an unconstrained quantity: the inlet
          residual then measures the GLOBAL mismatch between upstream supply
          and downstream draw, which Newton can only close through slow
          network-wide coupling, and whose local derivative does not match the
          Jacobian diagonal assembled from the adjacent pipelines.  That is a
          linear crawl, not convergence.  Taking the outlet-side draw instead
          makes the inlet balance a genuine equation in p_inlet whose
          derivative IS the pipeline diagonal, restoring quadratic behaviour.
        * Inlet is a reference node: mirror image, take the inlet side.
        * Both are junctions: neither balance is free, the symmetric average is
          the right compromise (original behaviour, unchanged).
        """
        inlet_is_reference = getattr(self.inlet, "node_type", None) == "reference"
        outlet_is_reference = getattr(self.outlet, "node_type", None) == "reference"

        if outlet_is_reference and not inlet_is_reference:
            # Deliver exactly what the downstream network draws from the
            # reference outlet; the reference node absorbs any discrepancy,
            # which is precisely what a reference node is for.
            self.flow_rate = total_flow_out + outlet_node_demand
        elif inlet_is_reference and not outlet_is_reference:
            self.flow_rate = total_flow_in - inlet_node_demand
        else:
            self.flow_rate = (
                total_flow_in - inlet_node_demand
                + total_flow_out + outlet_node_demand
            ) / 2


        # Convert to mass flow rate
        std_density = self.gas_mixture.standard_density if hasattr(self.gas_mixture, 'standard_density') else 0.8  # kg/sm3
        self.mass_flow_rate = self.flow_rate * std_density

        # Pressure constraint is handled in network.update_compressor_parameters()

    def calculate_power(self):
        """
        Calculate the power consumption of the compressor.
        Returns power consumption in MW.
        """
        if self.flow_rate is None or self.flow_rate == 0:
            self.power_consumption_value = 0.0
            return self.power_consumption_value
            
        T = self.T1  # Inlet temperature
        Q = abs(self.flow_rate)  # Volumetric flow rate
        
        # Power calculation using thermodynamic relations
        self.power_consumption_value = (
            self.cp * T * (self.compression_ratio ** ((self.n - 1) / self.n) - 1) * 
            self.mass_flow_rate / self.efficiency / 1e6  # Convert to MW
        )
        
        return self.power_consumption_value
    
    def power_consumption(self):
        """Legacy method name for backward compatibility."""
        return self.calculate_power()

    def calc_pipe_slope_correction(self):
        """Compressors don't have slope correction."""
        return 0
    
    def update_gas_mixture(self):
        """Update gas mixture based on flow direction."""
        if self.flow_rate is None or self.flow_rate >= 0:
            self.gas_mixture = self.inlet.gas_mixture
        else:
            self.gas_mixture = self.outlet.gas_mixture

    def calc_flow_rate(self):
        """Return the calculated flow rate."""
        return self.flow_rate if self.flow_rate is not None else 0.0


    def calculate_incoming_flows_and_derivatives(self, pipelines):
        """Calculate total flows and derivatives from connected pipelines."""
        total_flow_in = 0.0
        total_derivative_in = 0.0
        total_flow_out = 0.0
        total_derivative_out = 0.0

        for pipeline in pipelines:
            if pipeline.outlet == self.inlet:
                # Pipeline flowing into compressor inlet
                flow = pipeline.calc_flow_rate()
                derivative = pipeline.flow_rate_first_order_derivative(is_inlet=False)
                total_flow_in += flow
                total_derivative_in += derivative
            elif pipeline.inlet == self.outlet:
                # Pipeline flowing out of compressor outlet
                flow = pipeline.calc_flow_rate()
                derivative = pipeline.flow_rate_first_order_derivative(is_inlet=True)
                total_flow_out += flow
                total_derivative_out += derivative

        return total_flow_in, total_derivative_in, total_flow_out, total_derivative_out

    def calculate_stable_flow_rate(self, tol=0.001):
        """
        Calculate stable flow rate for compressor.
        For compressors, this is typically determined by mass balance.
        """
        return self.calc_flow_rate()

    def calc_gas_mass_flow(self):
        """Calculate gas mass flow rate through the compressor."""
        q = self.calc_flow_rate()
        std_density = self.gas_mixture.standard_density if hasattr(self.gas_mixture, 'standard_density') else 0.8  # kg/sm3
        return q * std_density

    def calc_flow_velocity(self):
        """Calculate flow velocity - not applicable for compressors."""
        return 0.0

