#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2025.
#     Developed by Yifei Lu
#     Last change on 1/2/25, 12:28 PM
#     Last change by yifei
#    *****************************************************************************

import numpy as np
from typing import Tuple
from scipy import sparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from collections import OrderedDict

# from .utils.gas_mixture.heating_value import *
from .utils.utils import *
from .node import *
from .pipeline import *
from .compressor import *
from .utils import *

# try:
#     import cupy as cp
#     import cupy.sparse.linalg as cpsplinalg
# except ImportError:
#     # logging.warning(f"CuPy is not installed or not available!")
#     print(f"CuPy is not installed or not available!")

from .utils.cuda_support import create_matrix_of_zeros, list_to_array
from .utils.initialization_strategies import INITIALIZATION_STRATEGIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Network:
    """
    Network class
    """

    def __init__(
        self,
        nodes: dict,
        pipelines=None,
        compressors=None,
        resistances=None,
        linear_resistances=None,
        shortpipes=None,
        run_initialization=True,
        pressure_prev=None,
        base_composition=None,
        run_composition_initialization=True,
        initialization_strategy='resistance_weighted',
        initialization_kwargs=None,
    ):
        """

        :param nodes:
        :param pipelines:
        """
        self.nodes = nodes
        self.pipelines = pipelines
        self.compressors = compressors
        self.resistances = resistances
        self.linear_resistances = linear_resistances
        self.shortpipes = shortpipes
        
        # Create index mapping for all components
        self._create_index_mappings()
        
        self.connections = self.all_edge_components()
        self._create_connection_mappings()  # Create connection mappings after connections are built
        self.connection_matrix = self.create_connection_matrix()
        self.reference_nodes = self.find_reference_nodes()
        self.non_junction_nodes = self.find_non_junction_nodes()
        self.junction_nodes = self.find_junction_nodes()
        self.run_initialization = run_initialization
        self.pressure_prev = pressure_prev
        self.run_composition_initialization = run_composition_initialization
        self.initialization_strategy = initialization_strategy
        self.initialization_kwargs = initialization_kwargs or {}

        if base_composition is not None:
            self.base_composition = base_composition
        else:
            self.base_composition = NATURAL_GAS_gri30
        # self.incidence_matrix = self.create_incidence_matrix()

    def _create_index_mappings(self):
        """
        Create index mappings between domain IDs and simulation indices for all components.
        
        Domain IDs: Original indices from CSV/user input (can be sparse like 1,5,10,15)
        Simulation indices: Sequential indices used in matrices/vectors (always 0,1,2,3)
        """
        # Node mapping: domain IDs → simulation node indices
        if not self.nodes:
            # Handle empty nodes case
            self._node_id_to_simulation_node_index = {}
            self._simulation_node_index_to_node_id = {}
            self._max_node_id = 0
            return
            
        sorted_node_ids = sorted(self.nodes.keys())
        self._node_id_to_simulation_node_index = {node_id: sim_idx for sim_idx, node_id in enumerate(sorted_node_ids)}
        self._simulation_node_index_to_node_id = {sim_idx: node_id for sim_idx, node_id in enumerate(sorted_node_ids)}
        self._max_node_id = max(self.nodes.keys())
        
        # Connection mapping will be created after connections are built
        self._connection_mappings_created = False
        
    def node_id_to_simulation_node_index(self, node_id):
        """
        Convert domain node ID to simulation node index for vectors/matrices.
        
        Args:
            node_id: Original node ID from CSV/user (e.g., 5, 10, 15)
        Returns:
            simulation_node_index: Sequential index for node vectors (e.g., 0, 1, 2)
        """
        if node_id not in self._node_id_to_simulation_node_index:
            raise ValueError(f"Node ID {node_id} not found in network")
        return self._node_id_to_simulation_node_index[node_id]
        
    def simulation_node_index_to_node_id(self, simulation_node_index):
        """
        Convert simulation node index to domain node ID.
        
        Args:
            simulation_node_index: Sequential index for node vectors (e.g., 0, 1, 2)
        Returns:
            node_id: Original node ID from CSV/user (e.g., 5, 10, 15)
        """
        if simulation_node_index not in self._simulation_node_index_to_node_id:
            raise ValueError(f"Simulation node index {simulation_node_index} not found in mapping")
        return self._simulation_node_index_to_node_id[simulation_node_index]
        
    def get_simulation_node_count(self):
        """
        Get the number of nodes for matrix/vector dimensions.
        """
        return len(self.nodes)

    def update_demands(self, csv_file: str):
        """
        Update the volumetric flow of all nodes using a new sf_nodes.csv file.
        This is used for sequential warm-start simulations (e.g., hourly).
        """
        import pandas as pd
        df = pd.read_csv(csv_file, sep=';')
        
        # Check if flow column exists
        flow_col = 'flow_sm3_per_s'
        if flow_col not in df.columns:
            return
            
        for _, row in df.iterrows():
            node_id = row['node_index']
            flow = row[flow_col]
            if pd.isna(flow):
                flow = 0.0
                
            if node_id in self.nodes:
                self.nodes[node_id].volumetric_flow = flow
                self.nodes[node_id].convert_volumetric_to_energy_flow()
                
    def _create_connection_mappings(self):

        """
        Create index mappings for connections after all edge components are combined.
        
        This maps:
        - Pipeline domain IDs (e.g., 1, 5, 10) → simulation edge indices (0, 1, 2, ...)
        - Resistance domain IDs (e.g., 2, 7) → simulation edge indices (3, 4, ...)
        - etc.
        """
        if not hasattr(self, 'connections') or self.connections is None:
            return
            
        # Build mappings for each component type
        self._pipeline_id_to_simulation_edge_index = {}
        self._resistance_id_to_simulation_edge_index = {}
        self._compressor_id_to_simulation_edge_index = {}
        self._linear_resistance_id_to_simulation_edge_index = {}
        self._shortpipe_id_to_simulation_edge_index = {}
        
        # Reverse mappings
        self._simulation_edge_index_to_component_info = {}  # sim_idx → (component_type, domain_id)
        
        for sim_edge_idx, connection in self.connections.items():
            component_type = type(connection).__name__
            
            if hasattr(connection, 'pipeline_index'):
                domain_id = connection.pipeline_index
                self._pipeline_id_to_simulation_edge_index[domain_id] = sim_edge_idx
                self._simulation_edge_index_to_component_info[sim_edge_idx] = ('Pipeline', domain_id)
            elif hasattr(connection, 'inlet_index') and component_type == 'Resistance':
                # Resistance doesn't have its own index, use a tuple of inlet/outlet
                domain_id = (connection.inlet_index, connection.outlet_index)
                self._resistance_id_to_simulation_edge_index[domain_id] = sim_edge_idx
                self._simulation_edge_index_to_component_info[sim_edge_idx] = ('Resistance', domain_id)
            # Add other component types as needed...
            
        self._connection_mappings_created = True
    
    def pipeline_id_to_simulation_edge_index(self, pipeline_id):
        """
        Convert pipeline domain ID to simulation edge index.
        
        Args:
            pipeline_id: Original pipeline ID from CSV/user (e.g., 5, 10, 15)
        Returns:
            simulation_edge_index: Sequential index for edge matrices (e.g., 0, 1, 2)
        """
        if not self._connection_mappings_created:
            self._create_connection_mappings()
        
        if pipeline_id not in self._pipeline_id_to_simulation_edge_index:
            raise ValueError(f"Pipeline ID {pipeline_id} not found in network")
        return self._pipeline_id_to_simulation_edge_index[pipeline_id]
    
    def simulation_edge_index_to_component_info(self, simulation_edge_index):
        """
        Convert simulation edge index to component information.
        
        Args:
            simulation_edge_index: Sequential index for edge matrices (e.g., 0, 1, 2)
        Returns:
            tuple: (component_type, domain_id) e.g., ('Pipeline', 5)
        """
        if not self._connection_mappings_created:
            self._create_connection_mappings()
            
        if simulation_edge_index not in self._simulation_edge_index_to_component_info:
            raise ValueError(f"Simulation edge index {simulation_edge_index} not found in mapping")
        return self._simulation_edge_index_to_component_info[simulation_edge_index]
    
    def get_simulation_edge_count(self):
        """
        Get the number of connections for matrix/vector dimensions.
        """
        return len(self.connections) if hasattr(self, 'connections') and self.connections else 0

    def all_edge_components(self):
        connections = dict()

        all_edge_classes = [
            self.pipelines,
            self.resistances,
            self.compressors,
            self.linear_resistances,
            self.shortpipes,
        ]

        i_connection = 0

        for edge_class in all_edge_classes:
            if edge_class is not None:
                for edge in edge_class.values():
                    connections[i_connection] = edge
                    i_connection += 1

        return connections

    def create_incidence_matrix(self):
        connections = self.all_edge_components()

        row_indices = []
        col_indices = []
        data_values = []

        for branch_id, branch_data in connections.items():
            inlet_dense_idx = self.node_id_to_simulation_node_index(branch_data.inlet_index)
            outlet_dense_idx = self.node_id_to_simulation_node_index(branch_data.outlet_index)
            
            row_indices.extend([inlet_dense_idx, outlet_dense_idx])
            col_indices.extend([branch_id, branch_id])  # branch_id starts from 0
            data_values.extend([1, -1])

        # Determine the shape of the incidence matrix
        num_nodes = self.get_simulation_node_count()
        num_branches = len(connections)
        shape = (num_nodes, num_branches)

        # Create the COO matrix
        incidence_matrix = coo_matrix(
            (data_values, (row_indices, col_indices)), shape=shape
        )
        return incidence_matrix

    def plot_pipeline_length_distribution(self):
        lines = self.pipelines.values()
        pipe_lengths = list()
        for l in lines:
            pipe_lengths.append(l.length / 1000)
        sns.histplot(data=pipe_lengths, stat="probability")
        plt.xlim((-2, max(pipe_lengths) + 10))
        plt.xlabel("Pipe length [km]")
        plt.show()
        return None

    def mapping_of_connections(self, use_cuda=False, sparse_matrix=False):
        n_nodes = self.get_simulation_node_count()
        mapping = create_matrix_of_zeros(
            n_nodes, use_cuda=use_cuda, sparse_matrix=sparse_matrix
        )
        for i_connection, connection in self.connections.items():
            i = self.node_id_to_simulation_node_index(connection.inlet_index)
            j = self.node_id_to_simulation_node_index(connection.outlet_index)
            mapping[i][j] = mapping[j][i] = i_connection
        return mapping

    def find_reference_nodes(self) -> list:
        """
        Find reference nodes, where pressure are pre-set
        :return: List of Node class of pressure-referenced nodes
        """
        pressure_ref_nodes = list()

        for node in self.nodes.values():
            # Check for reference node type, not just pressure presence
            if node.node_type == "reference" and node.pressure is not None:
                pressure_ref_nodes.append(node.index)

        return pressure_ref_nodes

    # def find_ref_nodes_index(self):
    #     """
    #     To get indices of the referenced nodes
    #     :return: List of pressure-referenced nodes indices and list of temperature-referenced nodes indices
    #     """
    #     pressure_ref_nodes_index = list()
    #     temperature_ref_nodes_index = list()
    #
    #     for node_index, node in self.nodes.items():
    #         if node.pressure is not None:
    #             pressure_ref_nodes_index.append(node_index)
    #         if node.temperature is not None:
    #             temperature_ref_nodes_index.append(node_index)
    #
    #     return pressure_ref_nodes_index, temperature_ref_nodes_index

    def demand_nodes_supply_pipelines(self):
        """
        Find supply pipelines for the demand nodes
        :return:
        """
        nodal_supply_pipelines = dict()

        if self.connections is not None:
            for i_connection, connection in self.connections.items():
                if connection.flow_rate is None or connection.flow_rate > 0:
                    if nodal_supply_pipelines.get(connection.outlet_index) is not None:
                        nodal_supply_pipelines[connection.outlet_index].append(
                            i_connection
                        )
                    else:
                        nodal_supply_pipelines[connection.outlet_index] = [i_connection]
                elif connection.flow_rate < 0:
                    if nodal_supply_pipelines.get(connection.inlet_index) is not None:
                        nodal_supply_pipelines[connection.inlet_index].append(
                            i_connection
                        )
                    else:
                        nodal_supply_pipelines[connection.inlet_index] = [i_connection]

        return OrderedDict(sorted(nodal_supply_pipelines.items()))

    # def convert_energy_flow_to_volumetric_flow(self, base='HHV'):
    #     for node in self.nodes.values():
    #         gas_comp = node.get_mole_fraction()
    #         standard_density = GasMixture(pressure=101325, temperature=288.15, composition=gas_comp).density
    #         LHV, HHV = calc_heating_value(node.gas_mixture)
    #         if base == 'HHV':
    #             heating_value = HHV/1e6*standard_density  # MJ/sm3
    #         elif base == 'LHV':
    #             heating_value = LHV/1e6*standard_density  # MJ/sm3
    #         else:
    #             raise ValueError
    #         if node.energy_flow is not None:
    #             node.volumetric_flow /= heating_value
    #             # try:
    #             #     h2_fraction = node.gas_mixture.zs[node.gas_mixture.components.index('hydrogen')]
    #             # except:
    #             #     h2_fraction = 0
    #             # node.flow /= (h2_fraction * 12.09 + (1-h2_fraction) * 38.28)
    #     return None

    def create_connection_matrix(self, use_cuda=False, sparse_matrix=False):
        n_nodes = self.get_simulation_node_count()
        pipelines = self.pipelines
        compressors = self.compressors
        resistances = self.resistances
        shortpipes = self.shortpipes
        if sparse_matrix:
            row_ind = list()
            col_ind = list()
            data = list()

        # Build a matrix to show the connection between nodes
        connection = create_matrix_of_zeros(
            n_nodes, use_cuda=use_cuda, sparse_matrix=sparse_matrix
        )

        if pipelines is not None:
            for pipe in pipelines.values():
                i = self.node_id_to_simulation_node_index(pipe.inlet_index)
                j = self.node_id_to_simulation_node_index(pipe.outlet_index)
                if sparse_matrix:
                    row_ind.append(i)
                    col_ind.append(j)
                    data.append(1)
                else:
                    connection[i][j] = 1
                    connection[j][i] = 1

        if compressors is not None:
            for compressor in compressors.values():
                i = self.node_id_to_simulation_node_index(compressor.inlet_index)
                j = self.node_id_to_simulation_node_index(compressor.outlet_index)
                if sparse_matrix:
                    row_ind.append(i)
                    col_ind.append(j)
                    data.append(2)
                else:
                    connection[i][j] = 2
                    connection[j][i] = 2

        if resistances is not None:
            for resistance in resistances.values():
                i = self.node_id_to_simulation_node_index(resistance.inlet_index)
                j = self.node_id_to_simulation_node_index(resistance.outlet_index)
                if sparse_matrix:
                    row_ind.append(i)
                    col_ind.append(j)
                    data.append(3)
                else:
                    connection[i][j] = 3
                    connection[j][i] = 3

        if shortpipes is not None:
            for sp in shortpipes.values():
                i = self.node_id_to_simulation_node_index(sp.inlet_index)
                j = self.node_id_to_simulation_node_index(sp.outlet_index)
                if sparse_matrix:
                    row_ind.append(i)
                    col_ind.append(j)
                    data.append(4)
                else:
                    connection[i][j] = 4
                    connection[j][i] = 4

        return connection

    def pressure_initialization(self):
        if self.initialization_strategy not in INITIALIZATION_STRATEGIES:
            raise ValueError(
                f"Unknown initialization strategy: '{self.initialization_strategy}'. "
                f"Available: {list(INITIALIZATION_STRATEGIES.keys())}"
            )
        pressure_init = INITIALIZATION_STRATEGIES[self.initialization_strategy](
            self, **self.initialization_kwargs
        )
        return self._propagate_through_compressors(pressure_init)

    def _propagate_through_compressors(self, pressure_init, num_passes: int = 8):
        """Estende l'inizializzazione attraverso gli archi compressore.

        PERCHE' SERVE. Le strategie di inizializzazione (resistance_weighted,
        che e' il default, e graph_based) camminano sulla rete seguendo
        `network.pipelines`. Un compressore NON e' una pipeline: e' un arco a
        se', quindi per loro non esiste. Tutta la sotto-rete che dipende da un
        compressore per raggiungere un nodo reference viene percio'
        inizializzata come se fosse scollegata dall'alimentazione.

        Finche' il nodo di mandata era un reference a pressione fissa il
        problema non si vedeva: la sotto-rete di valle partiva da li'. Con la
        formulazione aumentata quel nodo e' una giunzione normale, e senza
        questa propagazione il punto di partenza e' molto lontano dalla
        soluzione: su run_231 le pressioni iniziali portavano Newton a chiedere
        passi da 112 bar e a stallare subito.

        Si applica lo stesso schema di compressor_aware_initialization: piu'
        passate smorzate, cosi configurazioni in serie o in anello si
        stabilizzano senza oscillare.
        """
        if not self.compressors:
            return pressure_init

        riferimenti = set(self.reference_nodes or [])

        for _ in range(num_passes):
            cambiato = False
            for compressor in self.compressors.values():
                # Un reference ha pressione imposta: non si tocca.
                if compressor.outlet_index in riferimenti:
                    continue
                try:
                    i = self.node_id_to_simulation_node_index(compressor.inlet_index)
                    j = self.node_id_to_simulation_node_index(compressor.outlet_index)
                except (ValueError, KeyError):
                    continue

                p_in = pressure_init[i]
                if p_in is None or not p_in:
                    continue

                modo = getattr(compressor, "control_mode", "ratio")
                target = getattr(compressor, "p_out_target", None)
                if modo == "pressure" and target:
                    atteso = float(target)
                else:
                    atteso = float(p_in) * float(compressor.compression_ratio)

                vecchio = pressure_init[j] or 0.0
                # Smorzamento come in compressor_aware_initialization.
                pressure_init[j] = 0.7 * atteso + 0.3 * vecchio
                if abs(atteso - vecchio) > 1000.0:  # > 0.01 bar
                    cambiato = True

            if not cambiato:
                break

        return pressure_init

    def composition_initialization(
        self,
        tracking_method="simple_mixing",
        time_step=3600,
        cached_batch_information=None,
    ):
        """
        Seed nodal compositions by one topological propagation pass over the
        initial flow-direction DAG. Skipped when
        ``run_composition_initialization`` is False or when the network has no
        connections.
        """
        if not self.run_composition_initialization:
            return None
        if self.connections is None or len(self.connections) == 0:
            return None

        from GasNetSim.simulation.formulations.composition_tracker import (
            CompositionTracker,
        )

        if cached_batch_information is None:
            if tracking_method == "batch_tracking" and self.pipelines is not None:
                cached_batch_information = {
                    i: (
                        pipeline.batch_location_history.copy(),
                        pipeline.composition_history.copy(),
                    )
                    for i, pipeline in self.pipelines.items()
                }
            else:
                cached_batch_information = {}

        for connection in self.connections.values():
            connection.inlet = self.nodes[connection.inlet_index]
            connection.outlet = self.nodes[connection.outlet_index]

        self.update_connection_flow_rate()

        tracker = CompositionTracker(
            network=self,
            tracking_method=tracking_method,
            time_step=time_step,
            cached_batch_information=cached_batch_information,
        )
        nodal_composition = tracker.update()

        for node in self.nodes.values():
            if node.index in self.non_junction_nodes:
                node.gas_mixture.eos_composition_tmp = (
                    node.gas_mixture.eos_composition.copy()
                )
            node.gas_mixture.eos_composition = node.gas_mixture.eos_composition_tmp.copy()
            node.gas_mixture.convert_eos_composition_to_dictionary()
            node.gas_mixture.pressure = node.pressure
            node.gas_mixture.temperature = node.temperature
            node.gas_mixture.update_gas_mixture()

        if self.pipelines is not None:
            self.update_pipeline_parameters()
        if self.resistances is not None:
            self.update_resistance_parameters()
        if self.compressors is not None:
            self.update_compressor_parameters()

        self.update_connection_flow_rate()
        return nodal_composition

    def _snapshot_nodal_eos_compositions(self):
        return {
            node_id: np.array(node.gas_mixture.eos_composition, copy=True)
            for node_id, node in self.nodes.items()
        }

    def _refresh_network_state_from_gas_mixtures(self):
        for node in self.nodes.values():
            node.gas_mixture.pressure = node.pressure
            node.gas_mixture.temperature = node.temperature
            node.gas_mixture.update_gas_mixture()

            if node.flow_type == "volumetric":
                if node.volumetric_flow is not None:
                    node.convert_volumetric_to_energy_flow()
            elif node.flow_type == "energy":
                if node.energy_flow is not None:
                    node.convert_energy_to_volumetric_flow()
            else:
                raise ValueError(
                    "Unknown flow type, can be only volumetric or energy!"
                )

        if self.pipelines is not None:
            self.update_pipeline_parameters()
        if self.resistances is not None:
            self.update_resistance_parameters()
        if self.compressors is not None:
            self.update_compressor_parameters()

        self.update_connection_flow_rate()

    def _apply_relaxed_nodal_compositions(
        self, previous_compositions, target_compositions, relaxation_factor
    ):
        if not 0 < relaxation_factor <= 1:
            raise ValueError("composition_relaxation_factor must be in (0, 1].")

        non_junction_nodes = set(self.non_junction_nodes)
        max_delta = 0.0

        for node_id, node in self.nodes.items():
            previous = np.array(previous_compositions[node_id], copy=True)
            target = np.array(target_compositions[node_id], copy=True)

            if node_id in non_junction_nodes or relaxation_factor == 1.0:
                relaxed = target
            else:
                relaxed = ((1 - relaxation_factor) * previous) + (
                    relaxation_factor * target
                )
                composition_sum = relaxed.sum()
                if composition_sum > 0:
                    relaxed = relaxed / composition_sum

            node.gas_mixture.eos_composition = relaxed.copy()
            node.gas_mixture.eos_composition_tmp = relaxed.copy()
            node.gas_mixture.convert_eos_composition_to_dictionary()

            max_delta = max(max_delta, float(np.max(np.abs(relaxed - previous))))

        self._refresh_network_state_from_gas_mixtures()
        return max_delta

    def newton_raphson_initialization(self, warm_start=False):
        """
        Initialization for NR-solver.
        If warm_start=True, it uses the current node pressures.
        Otherwise, nodal pressures are initialized as 0.98 of inlet pressure.
        """

        nodes = self.nodes
        pipelines = self.pipelines
        resistances = self.resistances

        n_nodes = len(nodes)

        # Build a matrix to show the connection between nodes
        connection = self.create_connection_matrix()

        # TODO consider the case where ref_nodes do not start with index 0
        p_ref_nodes = self.reference_nodes

        nodal_flow_init = [
            x.volumetric_flow if x.volumetric_flow is not None else 0
            for x in nodes.values()
        ]
        
        # If warm start, we just take the current pressure instead of recalculating
        if warm_start:
            pressure_init = [x.pressure for x in nodes.values()]
        else:
            pressure_init = [x.pressure for x in nodes.values()]

        # TODO use ambient temperature to initialize outlet temperature
        temperature_init = [
            x.temperature if x.temperature is not None else 288.15
            for x in self.nodes.values()
        ]

        total_flow = sum([x for x in nodal_flow_init if x is not None])

        for n in p_ref_nodes:
            sim_node_idx = self.node_id_to_simulation_node_index(n)
            nodal_flow_init[sim_node_idx] = -total_flow / len(p_ref_nodes)

        if not warm_start:
            if self.run_initialization:
                pressure_init = self.pressure_initialization()
            else:
                pressure_init = self.pressure_prev

        for i in range(len(nodal_flow_init)):
            node_id = self.simulation_node_index_to_node_id(i)
            nodes[node_id].pressure = pressure_init[i]
            nodes[node_id].volumetric_flow = nodal_flow_init[i]
            nodes[node_id].temperature = temperature_init[i]
            if nodes[node_id].flow_type == "volumetric":
                nodes[node_id].convert_volumetric_to_energy_flow()
            elif nodes[node_id].flow_type == "energy":
                nodes[node_id].convert_energy_to_volumetric_flow()
            else:
                raise (
                    ValueError("Unknown flow type, can be only volumetric or energy!")
                )

        if pipelines is not None:
            for index, pipe in pipelines.items():
                pipe.inlet = nodes[pipe.inlet_index]
                pipe.outlet = nodes[pipe.outlet_index]

        if resistances is not None:
            for index, r in resistances.items():
                r.inlet = nodes[r.inlet_index]
                r.outlet = nodes[r.outlet_index]

        return nodal_flow_init, pressure_init, temperature_init

    def jacobian_matrix(self, use_cuda=False, sparse_matrix=False):

        connections = self.connections
        nodes = self.nodes

        non_junction_nodes_sim_indices = [self.node_id_to_simulation_node_index(x) for x in self.non_junction_nodes]

        n_nodes = self.get_simulation_node_count()

        n_junction_nodes = len(self.junction_nodes)

        jacobian_mat = create_matrix_of_zeros(
            n_nodes, use_cuda=use_cuda, sparse_matrix=sparse_matrix
        )
        flow_mat = create_matrix_of_zeros(
            n_nodes, use_cuda=use_cuda, sparse_matrix=sparse_matrix
        )

        for connection in connections.values():
            i = self.node_id_to_simulation_node_index(connection.inlet_index)
            j = self.node_id_to_simulation_node_index(connection.outlet_index)

            connection.calculate_stable_flow_rate()

            flow_mat[i][j] -= connection.flow_rate
            flow_mat[j][i] += connection.flow_rate

            if type(connection) is not ShortPipe:
                if type(connection) is Compressor:
                    # NOTE: no Jacobian contribution from the compressor.
                    #
                    # The previous implementation called
                    # calculate_incoming_flows_and_derivatives() and added
                    # total_derivative_in / total_derivative_out here.  Those
                    # quantities are sums of the FIRST-ORDER DERIVATIVES OF THE
                    # ADJACENT PIPELINES, which the pipeline branch below has
                    # already accumulated into the very same diagonal entries —
                    # with the opposite sign.  For every pipe p feeding the
                    # compressor inlet the pipeline branch writes
                    #     jacobian[i][i] += -p.deriv(is_inlet=False)
                    # and the compressor block wrote
                    #     jacobian[i][i] += +p.deriv(is_inlet=False)
                    # so the two cancelled exactly, wiping the inlet node's
                    # diagonal and leaving a near-singular row.  The resulting
                    # Newton direction is not a descent direction, the line
                    # search cannot find any step that reduces the residual,
                    # and the solver stalls.  (The block also lacked the
                    # non_junction_nodes_sim_indices guards used below.)
                    #
                    # A pressure-controlled compressor holds its outlet at the
                    # target pressure and passes flow freely, so d(flow)/dp at
                    # the inlet is ~0 to first order: contributing nothing is
                    # both simpler and closer to the truth.  The compressor
                    # still enters the RESIDUAL through flow_mat above, so this
                    # is a quasi-Newton treatment — slightly slower asymptotic
                    # convergence, but a consistent, non-singular Jacobian.
                    pass
                else:
                    # Standard pipeline/resistance handling
                    # NB: rimossa la riga morta
                    #     tmp = (abs(p1**2 - p2**2 - slope_corr)) ** (-0.5)
                    # 'tmp' non veniva mai usata (la derivata reale e' in
                    # flow_rate_first_order_derivative, gia' regolarizzata con
                    # _DELTA_P2_MIN). Con flusso ~zero (p1≈p2) la base andava a 0
                    # e 0**(-0.5)=inf generava il RuntimeWarning "divide by zero
                    # in scalar power": rumore innocuo, ora eliminato.
                    if i not in non_junction_nodes_sim_indices and j not in non_junction_nodes_sim_indices:
                        jacobian_mat[i][j] += connection.flow_rate_first_order_derivative(
                            is_inlet=False
                        )
                        jacobian_mat[j][i] += connection.flow_rate_first_order_derivative(
                            is_inlet=True
                        )
                    if i not in non_junction_nodes_sim_indices:
                        jacobian_mat[i][i] += -connection.flow_rate_first_order_derivative(
                            is_inlet=True
                        )
                    if j not in non_junction_nodes_sim_indices:
                        jacobian_mat[j][j] += -connection.flow_rate_first_order_derivative(
                            is_inlet=False
                        )

        jacobian_mat = delete_matrix_rows_and_columns(jacobian_mat, non_junction_nodes_sim_indices)
        # flow_mat = delete_matrix_rows_and_columns(flow_mat, non_junction_nodes_dense)

        return jacobian_mat, flow_mat

    def find_non_junction_nodes(self):
        non_junction_nodes = list()

        if self.shortpipes is not None:
            for sp in self.shortpipes.values():
                non_junction_nodes.append(sp.inlet_index)
        for node in self.nodes.values():
            if node.node_type == "reference":
                non_junction_nodes.append(node.index)

        return sorted(non_junction_nodes)

    def find_junction_nodes(self):
        return [
            node.index
            for node in self.nodes.values()
            if node.index not in self.non_junction_nodes
        ]

    def newton_raphson_iteration(self, target):
        jacobian_matrix, flow_matrix = self.jacobian_matrix()
        number_of_junction_nodes = len(jacobian_matrix)
        j_mat_inv = np.linalg.inv(jacobian_matrix)

        # Flow balance of connecting pipeline elements at all nodes
        delta_flow = target - np.dot(flow_matrix, np.ones(number_of_junction_nodes))

        # Select only the flows at junction nodes
        delta_flow = [
            delta_flow[i]
            for i in range(len(delta_flow))
            if self.simulation_node_index_to_node_id(i) not in self.junction_nodes
        ]

    def calculate_nodal_inflow_composition(self):
        pass

    def update_node_parameters(self, pressure, flow, temperature):
        for i in range(len(flow)):
            node_id = self.simulation_node_index_to_node_id(i)
            self.nodes[node_id].pressure = pressure[i]
            self.nodes[node_id].volumetric_flow = flow[i]
            self.nodes[node_id].gas_mixture.pressure = self.nodes[node_id].pressure
            self.nodes[node_id].gas_mixture.temperature = self.nodes[node_id].temperature
            self.nodes[node_id].gas_mixture.update_gas_mixture()
            # self.nodes[node_id].update_gas_mixture()

            if self.nodes[node_id].flow_type == "volumetric":
                self.nodes[node_id].convert_volumetric_to_energy_flow()
            elif self.nodes[node_id].flow_type == "energy":
                self.nodes[node_id].convert_energy_to_volumetric_flow()
            else:
                raise (
                    ValueError("Unknown flow type, can be only volumetric or energy!")
                )

    # def simulation(self, composition_tracking=False):
    #     logging.debug([x.flow for x in self.nodes.values()])
    #     # ref_nodes = self.p_ref_nodes_index
    #
    #     n_nodes = len(self.nodes.keys())
    #     n_non_junction_nodes = len(self.non_junction_nodes)
    #     connection_matrix = self.connection_matrix
    #
    #     init_f, init_p, init_t = self.newton_raphson_initialization()
    #
    #     max_iter = 100
    #     n_iter = 0
    #     # n_non_ref_nodes = n_nodes - len(ref_nodes)
    #
    #     f = np.array(init_f)
    #     p = np.array(init_p)
    #     t = np.array(init_t)
    #     logging.info(f'Initial pressure: {p}')
    #     logging.info(f'Initial flow: {f}')
    #
    #     # OLD CODE - FIXED: Must use simulation node indexing
    #     # for i in range(len(init_f)):
    #     #     node_id = self.simulation_node_index_to_node_id(i)
    #     #     self.nodes[node_id].pressure = pressure[i]
    #     #     self.nodes[node_id].volumetric_flow = flow[i]
    #     #     self.nodes[node_id].convert_volumetric_to_energy_flow()
    #     #     self.nodes[node_id].update_gas_mixture()

    def update_pipeline_parameters(self):
        for index, pipe in self.pipelines.items():
            pipe.inlet = self.nodes[pipe.inlet_index]
            pipe.outlet = self.nodes[pipe.outlet_index]
            pipe.update_gas_mixture()

    def update_resistance_parameters(self):
        for index, r in self.resistances.items():
            r.inlet = self.nodes[r.inlet_index]
            r.outlet = self.nodes[r.outlet_index]
            r.update_gas_mixture()

    def update_compressor_parameters(self):
        """Update compressor parameters and calculate flow rates."""
        if self.compressors is not None:
            # Quando la formulazione aumentata e attiva, la portata di ogni
            # compressore e una VERA INCOGNITA del sistema (vedi
            # PressureProblem con augment_compressors=True): viene scritta in
            # compressor.flow_rate a partire da x prima di questa chiamata.
            #
            # In quel caso qui non si deve fare nulla di piu che riallineare i
            # riferimenti ai nodi e la miscela. In particolare NON si deve:
            #  * ricalcolare flow_rate con l'euristica della media, perche
            #    sovrascriverebbe l'incognita;
            #  * sovrascrivere la pressione del nodo di uscita, perche quello
            #    e' esattamente il difetto che rende il residuo non funzione
            #    di x (passi da centinaia di bar annullati dalla riscrittura).
            #    Il legame p_out = ratio * p_in e' ora imposto da una riga di
            #    vincolo del sistema, non da un'assegnazione.
            augmented = getattr(self, "compressor_flow_from_state", False)

            for index, compressor in self.compressors.items():
                compressor.inlet = self.nodes[compressor.inlet_index]
                compressor.outlet = self.nodes[compressor.outlet_index]
                compressor.update_gas_mixture()

                if augmented:
                    continue

                # Calculate compressor flow rates from connected pipelines
                if self.pipelines is not None:
                    total_flow_in, total_derivative_in, total_flow_out, total_derivative_out = \
                        compressor.calculate_incoming_flows_and_derivatives(self.pipelines.values())
                    
                    # Fix sign convention: node demands should be positive for consumption
                    inlet_node_demand = compressor.inlet.volumetric_flow if compressor.inlet.volumetric_flow is not None else 0.0
                    outlet_node_demand = compressor.outlet.volumetric_flow if compressor.outlet.volumetric_flow is not None else 0.0
                    
                    compressor.update_flow_rate(total_flow_in, total_flow_out, inlet_node_demand, outlet_node_demand)
                    
                    # Enforce compressor pressure constraint
                    if getattr(compressor, 'control_mode', 'ratio') == 'pressure':
                        if getattr(compressor, 'p_out_target', None) is not None:
                            # In PCM, the outlet pressure is fixed by the Reference node.
                            # We just update the ratio for power consumption calculations.
                            inlet_p = max(self.nodes[compressor.inlet_index].pressure, 1.0)
                            compressor.compression_ratio = max(compressor.p_out_target / inlet_p, 1.0)
                    else:
                        self.nodes[compressor.outlet_index].pressure = self.nodes[compressor.inlet_index].pressure * compressor.compression_ratio

    def update_connection_flow_rate(self):
        for connection in self.connections.values():
            connection.flow_rate = connection.calc_flow_rate()
            connection.mass_flow_rate = connection.calc_gas_mass_flow()
            connection.flow_velocity = connection.calc_flow_velocity()

    def fun(self, p):
        init_f, init_p, init_t = self.newton_raphson_initialization()
        init_f = [
            init_f[i]
            for i in range(len(init_f))
            if self.simulation_node_index_to_node_id(i) not in self.non_junction_nodes
        ]

        f_target = np.array(init_f)

        flow_vector = calculate_flow_vector(
            network=self, pressure_bar=p, target_flow=f_target
        )
        # print(sorted([abs(x) for x in flow_vector])[-10:])

        return flow_vector

    def solving(self, fun, x0, jac, method, tol):
        sol = optimize.root(fun, x0, jac=jac, method=method, tol=tol)
        return sol.x

    def save_pressure_values(self):
        return [n.pressure for n in self.nodes.values()]

    def assign_pressure_values(self, p):
        for i in self.nodes.keys():
            if i not in self.reference_nodes:
                sim_idx = self.node_id_to_simulation_node_index(i)
                self.nodes[i].pressure = p[sim_idx]  # update nodal pressure

    def newton_raphson_solving(
        self, fun, jac, x, target, alpha=1.0, tol=0.001, max_iter=100
    ):
        err = 1.0 + tol  # ensure the first iteration will be performed
        n_iter = 0

        # For the first iteration
        delta_q = fun(x)

        while err > tol:
            dq_dp = -jac(x)
            delta_p = np.linalg.solve(dq_dp, delta_q)
            x += delta_p / alpha  # applying the under-relaxation factor to delta_p
            delta_q = fun(x)  # update delta_q
            err = abs(delta_q).max()
            n_iter += 1
            if n_iter >= max_iter:
                return False

        return x

    def simulation(
        self,
        max_iter=100,
        tol=0.001,
        underrelaxation_factor=2.0,
        use_cuda=False,
        sparse_matrix=False,
        tracking_method="simple_mixing",
        time_step=3600,
        problem=None,
        solver=None,
        coupling_strategy="direct",
        coupling_max_iter=20,
        coupling_tol=1e-4,
        outer_pressure_tol=None,
        composition_relaxation_factor=0.5,
        warm_start=False,
    ):
        from GasNetSim.simulation.strategies import DirectStrategy, StaggeredStrategy

        common = dict(
            max_iter=max_iter,
            tol=tol,
            underrelaxation_factor=underrelaxation_factor,
            use_cuda=use_cuda,
            sparse_matrix=sparse_matrix,
            tracking_method=tracking_method,
            time_step=time_step,
            warm_start=warm_start,
        )

        if coupling_strategy == "direct":
            return DirectStrategy(problem=problem, solver=solver).solve(
                self, **common
            )

        if coupling_strategy in ("staggered", "outer_picard"):
            if problem is not None:
                raise ValueError(
                    "Custom problem instances are only supported with "
                    "coupling_strategy='direct'."
                )
            return StaggeredStrategy(
                solver=solver,
                coupling_max_iter=coupling_max_iter,
                coupling_tol=coupling_tol,
                outer_pressure_tol=outer_pressure_tol,
                composition_relaxation_factor=composition_relaxation_factor,
            ).solve(self, **common)

        raise ValueError(
            f"Unknown coupling_strategy: {coupling_strategy!r}. "
            "Available: ['direct', 'staggered']"
        )
