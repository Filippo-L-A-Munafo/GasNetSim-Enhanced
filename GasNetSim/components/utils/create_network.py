#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 12/19/24, 10:24 AM
#     Last change by yifei
#    *****************************************************************************
from collections import OrderedDict
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import warnings
from scipy.constants import atm

from ..network import Network
from ..node import Node
from ..pipeline import Pipeline, Resistance, ShortPipe, LinearResistance
from ..compressor import Compressor
from ..gas_mixture.typical_mixture_composition import COMMON_GAS_COMPOSITIONS
from ...utils.exception import *

AVAILABLE_GAS_NAMES = ", ".join(COMMON_GAS_COMPOSITIONS.keys())

def get_builtin_gas_composition(name):
    if name not in COMMON_GAS_COMPOSITIONS:
        raise ValueError(
            f"The gas mixture '{name}' is not implemented.\n"
            f"Available gas mixtures: {AVAILABLE_GAS_NAMES}\n"
            f"If you want to use a specific gas mixture composition, please assign it as a OrderedDict or implement it separately."
        )
    return COMMON_GAS_COMPOSITIONS[name]


def convert_gas_composition(gas_composition: str) -> OrderedDict:
    try:
        return OrderedDict(eval(gas_composition))
    except (SyntaxError, ValueError):
        raise ValueError(
            f"Invalid gas composition provided: {gas_composition}."
            f"Please provide a valid built-in name or a correctly formatted dictionary string."
        )


def read_nodes(path_to_file: Path, base_composition=None) -> dict[int, Node]:
    """
    Read nodes from a CSV file and create Node objects.

    :param path_to_file: Path to the CSV file containing nodes information.
    :return: A dictionary of node indices to Node objects.
    """
    nodes = {}
    df_node = pd.read_csv(path_to_file, delimiter=";")
    df_node = df_node.replace({np.nan: None})

    if base_composition is None:
        # Infer the default from the first reference node so demand nodes start
        # with gas properties consistent with the actual upstream supply.
        for _, row in df_node.iterrows():
            if (
                row.get("node_type") == "reference"
                and row["gas_composition"] is not None
            ):
                gas_composition_str = row["gas_composition"]
                if "{" not in gas_composition_str and "}" not in gas_composition_str:
                    base_composition = get_builtin_gas_composition(
                        gas_composition_str
                    )
                else:
                    base_composition = convert_gas_composition(gas_composition_str)
                break
        if base_composition is None:
            base_composition = COMMON_GAS_COMPOSITIONS["NATURAL_GAS_gri30"]

    for _, row in df_node.iterrows():
        # Convert gauge pressure to absolute pressure
        gauge_pressure_pa = row["pressure_pa"]
        absolute_pressure_pa = gauge_pressure_pa + atm if gauge_pressure_pa is not None else None

        if row["gas_composition"] is not None:
            gas_composition_str = row["gas_composition"]

            if "{" not in gas_composition_str and "}" not in gas_composition_str:
                # Assume it's a built-in gas composition name
                row["gas_composition"] = get_builtin_gas_composition(gas_composition_str)
            else:
                # Assume it's a string representation of a dictionary of gas composition
                row["gas_composition"] = convert_gas_composition(row["gas_composition"])
        else:
            row["gas_composition"] = base_composition

        nodes[row["node_index"]] = Node(
            node_index=row["node_index"],
            pressure_pa=absolute_pressure_pa,
            volumetric_flow=row["flow_sm3_per_s"],
            energy_flow=row["flow_MW"],
            temperature=row["temperature_k"],
            altitude=row["altitude_m"],
            gas_composition=row["gas_composition"],
            node_type=row["node_type"],
            flow_type=row["flow_type"],
            longitude=row.get("longitude"),
            latitude=row.get("latitude"),
        )
    return nodes


def read_pipelines(
    path_to_file: Path, network_nodes: dict, conversion_factor=1.0
) -> dict:
    """

    :param path_to_file:
    :param network_nodes:
    :return:
    """
    pipelines = dict()
    df_pipe = pd.read_csv(path_to_file, delimiter=";")
    df_pipe = df_pipe.replace({np.nan: None})

    for row_index, row in df_pipe.iterrows():
        friction_method = row.get("friction_method", "chen") or "chen"
        pipelines[row["pipeline_index"]] = Pipeline(
            pipeline_index=row["pipeline_index"],
            inlet=network_nodes[row["inlet_index"]],
            outlet=network_nodes[row["outlet_index"]],
            diameter=row["diameter_m"],
            length=row["length_m"],
            friction_factor_method=friction_method,
            conversion_factor=conversion_factor,
        )
    return pipelines


def read_compressors(path_to_file: Path, network_nodes: dict) -> dict:
    """
    Read compressors from a CSV file and create Compressor objects.

    :param path_to_file: Path to the CSV file containing compressor information.
    :param network_nodes: Dictionary of existing network nodes.
    :return: A dictionary of compressor indices to Compressor objects.
    """
    compressors = dict()
    
    try:
        df_compressors = pd.read_csv(path_to_file, delimiter=";")
        df_compressors = df_compressors.replace({np.nan: None})
        
        for index, row in df_compressors.iterrows():
            compressor_index = int(row["compressor_index"])
            inlet_index = int(row["inlet"])
            outlet_index = int(row["outlet"])
            compression_ratio = float(row["compression_ratio"]) if row["compression_ratio"] is not None else 1.1
            efficiency = float(row["efficiency"]) if row["efficiency"] is not None else 0.85
            thermodynamic_process = row["thermodynamic_process"] if row["thermodynamic_process"] is not None else "isentropic"
            drive = row["drive"] if "drive" in row and row["drive"] is not None else "electric"

            # Modalita' di controllo e pressione di mandata.
            #
            # Queste due colonne venivano IGNORATE: il Compressor nasceva
            # sempre con i default del costruttore (control_mode="ratio",
            # p_out_target=None), quindi un CSV che chiedeva
            # control_mode=pressure con p_out_target=7e6 veniva risolto come
            # se fosse a rapporto fisso. Il target di mandata non arrivava mai
            # al solver, e le due modalita' producevano risultati identici.
            control_mode = "ratio"
            if "control_mode" in row and row["control_mode"] is not None:
                valore = str(row["control_mode"]).strip().lower()
                if valore in ("ratio", "pressure"):
                    control_mode = valore

            p_out_target = None
            if "p_out_target" in row and row["p_out_target"] is not None:
                try:
                    grezzo = float(row["p_out_target"])
                    # Le celle vuote diventano NaN -> None sopra, ma un CSV
                    # scritto a mano puo' contenere 0: un target nullo non ha
                    # senso fisico e va trattato come "non specificato".
                    if grezzo > 0:
                        p_out_target = grezzo
                except (TypeError, ValueError):
                    p_out_target = None

            # Un controllo di pressione senza target non e' realizzabile:
            # si ricade sul rapporto invece di fallire silenziosamente.
            if control_mode == "pressure" and p_out_target is None:
                logging.warning(
                    "Compressore %s: control_mode='pressure' ma p_out_target "
                    "assente o non valido. Uso il controllo di rapporto.",
                    compressor_index,
                )
                control_mode = "ratio"

            # Get inlet and outlet nodes
            inlet_node = network_nodes[inlet_index]
            outlet_node = network_nodes[outlet_index]

            # Create compressor object
            compressor = Compressor(
                compressor_index=compressor_index,
                inlet=inlet_node,
                outlet=outlet_node,
                compression_ratio=compression_ratio,
                efficiency=efficiency,
                thermodynamic_process=thermodynamic_process,
                drive=drive,
                control_mode=control_mode,
                p_out_target=p_out_target,
            )
            
            compressors[compressor_index] = compressor
            
    except FileNotFoundError:
        print(f"Compressor file not found: {path_to_file}")
    except Exception as e:
        print(f"Error reading compressors: {e}")
        
    return compressors


def read_resistances(path_to_file: Path, network_nodes: dict) -> dict:
    """

    :param path_to_file:
    :param network_nodes:
    :return:
    """
    resistances = dict()
    df_resistance = pd.read_csv(path_to_file, delimiter=";")
    df_resistance = df_resistance.replace({np.nan: None})

    for row_index, row in df_resistance.iterrows():
        resistances[row["resistance_index"]] = Resistance(
            inlet=network_nodes[row["inlet_index"]],
            outlet=network_nodes[row["outlet_index"]],
            resistance=row["resistance"],
        )
    return resistances


def read_linear_resistances(path_to_file: Path, network_nodes: dict) -> dict:
    """

    :param path_to_file:
    :param network_nodes:
    :return:
    """
    resistances = dict()
    df_linear_resistance = pd.read_csv(path_to_file, delimiter=";")
    df_linear_resistance = df_linear_resistance.replace({np.nan: None})

    for row_index, row in df_linear_resistance.iterrows():
        resistances[row["linear_resistance_index"]] = LinearResistance(
            inlet=network_nodes[row["inlet_index"]],
            outlet=network_nodes[row["outlet_index"]],
            resistance=row["linear_resistance"],
        )
    return resistances


def read_shortpipes(path_to_file: Path, network_nodes: dict) -> dict:
    """

    :param path_to_file:
    :param network_nodes:
    :return:
    """
    shortpipes = dict()
    df_shortpipes = pd.read_csv(path_to_file, delimiter=";")
    df_shortpipes = df_shortpipes.replace({np.nan: None})

    for row_index, row in df_shortpipes.iterrows():
        shortpipes[row["shortpipe_index"]] = ShortPipe(
            inlet=network_nodes[row["inlet_index"]],
            outlet=network_nodes[row["outlet_index"]],
        )
    return shortpipes


import warnings


def create_network_from_csv(
    path_to_folder: Path,
    conversion_factor=1.0,
    base_composition=None,
    **network_kwargs,
) -> Network:
    """
    Create a Network object from CSV files located in the specified folder.

    :param path_to_folder: Path to the folder containing the CSV files.
    :param conversion_factor: Conversion factor for pipeline data.
    :return: A Network object.
    """
    warnings.warn(
        "create_network_from_csv() is deprecated and will be removed in a future version. "
        "Please use create_network_from_folder() instead.",
        FutureWarning,
        stacklevel=2,
    )
    return create_network_from_folder(
        path_to_folder,
        conversion_factor,
        base_composition,
        **network_kwargs,
    )


def create_network_from_folder(
    path_to_folder: Path,
    conversion_factor=1.0,
    base_composition=None,
    **network_kwargs,
) -> Network:
    """
    Create a Network object from CSV files located in the specified folder.

    :param path_to_folder: Path to the folder containing the CSV files.
    :param conversion_factor: Conversion factor for pipeline data.
    :return: A Network object.
    """
    all_files = list(path_to_folder.glob("*.csv"))
    nodes_file = next((file for file in all_files if "node" in file.stem), None)

    if nodes_file is None:
        raise FileNotFoundError("Nodes file is required to create the network.")

    nodes = read_nodes(nodes_file, base_composition=base_composition)

    # Initialize network components
    network_components = {
        "nodes": nodes,
        "pipelines": None,
        "compressors": None,
        "resistances": None,
        "shortpipes": None,
        "linear_resistances": None,
    }

    # Mapping of component names to their corresponding read functions
    read_functions = {
        "pipeline": read_pipelines,
        "compressor": read_compressors,
        "resistance": read_resistances,
        "linearR": read_linear_resistances,
        "shortpipe": read_shortpipes,
    }

    # Read other components if provided
    for file in all_files:
        file_name = file.stem
        for component_key, read_function in read_functions.items():
            if component_key in file_name:
                if component_key == "pipeline":
                    network_components[component_key + "s"] = read_function(
                        file, nodes, conversion_factor
                    )
                else:
                    network_components[component_key + "s"] = read_function(file, nodes)
                break

    # Create and return the Network object
    return Network(
        nodes=network_components["nodes"],
        pipelines=network_components["pipelines"],
        compressors=network_components["compressors"],
        resistances=network_components["resistances"],
        linear_resistances=network_components["linear_resistances"],
        shortpipes=network_components["shortpipes"],
        **network_kwargs,
    )


def create_network_from_files(
    component_files: dict[str, Path],
    conversion_factor=1.0,
    **network_kwargs,
) -> Network:
    """
    Create a Network object from specified component CSV files.

    :param component_files: A dictionary mapping component names (e.g., 'nodes', 'pipelines') to file paths.
    :param conversion_factor: Conversion factor for pipeline data.
    :return: A Network object.
    """
    # Ensure nodes file is provided
    nodes_file = component_files.get("nodes")
    if nodes_file is None:
        raise ValueError("Nodes file is required to create the network.")

    # Read nodes
    nodes = read_nodes(nodes_file)

    # Initialize network components
    network_components = {
        "nodes": nodes,
        "pipelines": None,
        "compressors": None,
        "resistances": None,
        "shortpipes": None,
        "linear_resistances": None,
    }

    # Mapping of component names to their corresponding read functions
    read_functions = {
        "pipelines": read_pipelines,
        "compressors": read_compressors,
        "resistances": read_resistances,
        "linear_resistances": read_linear_resistances,
        "shortpipes": read_shortpipes,
    }

    # Read other components if provided
    for component_name, read_function in read_functions.items():
        if component_name in component_files:
            if component_name == "pipelines":
                network_components[component_name] = read_function(
                    component_files[component_name], nodes, conversion_factor
                )
            else:
                network_components[component_name] = read_function(
                    component_files[component_name], nodes
                )

    # Create and return the Network object
    return Network(
        nodes=network_components["nodes"],
        pipelines=network_components["pipelines"],
        compressors=network_components["compressors"],
        resistances=network_components["resistances"],
        linear_resistances=network_components["linear_resistances"],
        shortpipes=network_components["shortpipes"],
        **network_kwargs,
    )
