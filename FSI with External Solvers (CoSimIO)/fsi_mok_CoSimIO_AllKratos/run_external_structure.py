import KratosMultiphysics as Kratos
from KratosMultiphysics.OptimizationApplication.utilities.logger_utilities import time_decorator
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.CoSimulationApplication import CoSimIO
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication

# External Libraries
import numpy as np


with open("ProjectParametersCSM.json",'r') as parameter_file:
    parameters = Kratos.Parameters(parameter_file.read())

model = Kratos.Model()
simulation = StructuralMechanicsAnalysis(model,parameters)
simulation.Initialize() # Remember the steps: Initialize, RunSolutionLoop and Finalize

# Now you can work with it
#simulation.Run()

### CoSIM Registers
s_connection_name = ""
structural_mesh_name = "structural_mesh"

# Create a VTU output
model_part = model["Structure"]
interface_sub_model_part = model_part.GetSubModelPart("interface")
Vtu = Kratos.VtuOutput(interface_sub_model_part)

def cosimio_check_equal(a, b):
    assert a == b

@time_decorator()
def AdvanceInTime(info):
    settings = CoSimIO.Info()
    settings.SetString("identifier", "AdvanceInTime")
    settings.SetString("connection_name", s_connection_name)

    simulation.time = simulation._AdvanceTime()
    return CoSimIO.Info()

@time_decorator()
def InitializeSolutionStep(info):
    settings = CoSimIO.Info()
    settings.SetString("connection_name", s_connection_name)
    settings.SetString("identifier", "InitializeSolutionStep")
    model_part = model["Structure"]
    #model_part.ProcessInfo[Kratos.DELTA_TIME] = 0.01
    simulation.InitializeSolutionStep()
    #CoSimIO.ExportInfo(settings)
    return CoSimIO.Info()

@time_decorator()
def Predict(info):
    settings = CoSimIO.Info()
    settings.SetString("connection_name", s_connection_name)
    settings.SetString("identifier", "Predict")
    simulation._GetSolver().Predict()
    #CoSimIO.ExportInfo(settings)
    return CoSimIO.Info()

@time_decorator()
def SolveSolutionStep(info):
    is_converged = simulation._GetSolver().SolveSolutionStep()
    simulation._AnalysisStage__CheckIfSolveSolutionStepReturnsAValue(is_converged)

    # model_part = model["Structure"]
    # interface_sub_model_part = model_part.GetSubModelPart("interface")
    # Nodal_exp = Kratos.Expression.NodalExpression(interface_sub_model_part)
    # Kratos.Expression.VariableExpressionIO.Read(Nodal_exp, StructuralMechanicsApplication.POINT_LOAD)
    # Vtu.AddContainerExpression("interface_point_load", Nodal_exp)
    # Vtu.PrintOutput("interface_point_load")
    
    return CoSimIO.Info()

@time_decorator()
def FinalizeSolutionStep(info):
    settings = CoSimIO.Info()
    settings.SetString("connection_name", s_connection_name)
    settings.SetString("identifier", "FinalizeSolutionStep")
    simulation.FinalizeSolutionStep()
    #CoSimIO.ExportInfo(settings)
    return CoSimIO.Info()

@time_decorator()
def OutputSolutionStep(info):
    settings = CoSimIO.Info()
    settings.SetString("connection_name", s_connection_name)
    settings.SetString("identifier", "OutputSolutionStep")
    simulation.OutputSolutionStep()
    #CoSimIO.ExportInfo(settings)
    return CoSimIO.Info()

@time_decorator()
def ImportData(info):
    settings = CoSimIO.Info()
    settings.SetString("connection_name", s_connection_name)
    settings.SetString("identifier", info.GetString("identifier"))

    # Import the point loads from the fluid solver to the structural solver
    imported_force_vector = CoSimIO.DoubleVector()
    CoSimIO.ImportData(settings, imported_force_vector)
    print("Force vector", imported_force_vector)
    data_size = len(imported_force_vector)

    imported_force_vector_numpy = np.arange(data_size, dtype=np.float64)
    for i, data in enumerate(imported_force_vector):
        imported_force_vector_numpy[i] = data
    num_forces = imported_force_vector_numpy.shape[0] // 3
    print("----------------------------------------------------------------------------")
    print(num_forces)

    # Apply the imported forces to the structural solver as POINT_LOAD
    model_part = model["Structure"]
    interface_sub_model_part = model_part.GetSubModelPart("interface")

    nodes = list(interface_sub_model_part.Nodes)
    if len(nodes) != num_forces:
        raise Exception(f"Mismatch: {len(nodes)} nodes vs {num_forces} forces")

    # Apply the forces as POINT_LOAD
    for i, node in enumerate(nodes):
        fx, fy, fz = imported_force_vector_numpy[i*3:(i+1)*3]
        node.SetSolutionStepValue(StructuralMechanicsApplication.POINT_LOAD, 0, [fx, fy, fz])

    return CoSimIO.Info()

@time_decorator()
def ExportData(info):
    settings = CoSimIO.Info()
    settings.SetString("connection_name", s_connection_name)
    settings.SetString("identifier", info.GetString("identifier"))

    # Access the interface submodel part
    interface_model_part = model["Structure"].GetSubModelPart("interface") 

    # Create a flat list of displacements
    displacement_vector = []
    displacement_vector.clear()
    for node in interface_model_part.Nodes:
        disp = node.GetSolutionStepValue(Kratos.DISPLACEMENT)
        displacement_vector.extend([disp[0], disp[1], disp[2]])

    print("----------------------------------------------------------------------------")
    print("Displacements", len(displacement_vector))

    # Convert to CoSimIO.DoubleVector
    data_to_send = CoSimIO.DoubleVector(displacement_vector)

    # Export via CoSimIO
    return_info = CoSimIO.ExportData(settings, data_to_send)

    return CoSimIO.Info()

@time_decorator()
def ImportMesh(info):
    settings = CoSimIO.Info()
    settings.SetString("connection_name", s_connection_name)
    settings.SetString("identifier", "info_for_test")
    settings.SetString("name_for_check", "ImportMesh")
    if (info.Has("identifier")):
        settings.SetString("identifier_control", info.GetString("identifier"))
    #CoSimIO.ExportInfo(settings)
    return CoSimIO.Info()

@time_decorator()
def ExportMesh(info):
    # Exporting mesh to Kratos
    info = CoSimIO.Info()
    info.SetString("identifier", structural_mesh_name)
    info.SetString("connection_name", s_connection_name)

   # Access the computing model part
    model_part = model["Structure"]
    interface_sub_model_part = model_part.GetSubModelPart("interface")
  
    return_info = CoSimIO.ExportMesh(info, interface_sub_model_part)
    return info


# Connection Settings
settings = CoSimIO.Info()
settings.SetString("my_name", "run_external_structure")
settings.SetString("connect_to", "external_structure")
settings.SetInt("echo_level", 1)
settings.SetString("version", "1.25")
settings.SetString("communication_format", "file") 

# Connecting
return_info = CoSimIO.Connect(settings)
cosimio_check_equal(return_info.GetInt("connection_status"), CoSimIO.ConnectionStatus.Connected)
s_connection_name = return_info.GetString("connection_name")

# registering the functions
fct_info = CoSimIO.Info()
fct_info.SetString("connection_name", s_connection_name)

fct_info.SetString("function_name", "AdvanceInTime")
CoSimIO.Register(fct_info,           AdvanceInTime)

fct_info.SetString("function_name", "InitializeSolutionStep")
CoSimIO.Register(fct_info,           InitializeSolutionStep)

fct_info.SetString("function_name", "Predict")
CoSimIO.Register(fct_info,           Predict)

fct_info.SetString("function_name", "SolveSolutionStep")
CoSimIO.Register(fct_info,           SolveSolutionStep)

fct_info.SetString("function_name", "FinalizeSolutionStep")
CoSimIO.Register(fct_info,           FinalizeSolutionStep)

fct_info.SetString("function_name", "OutputSolutionStep")
CoSimIO.Register(fct_info,           OutputSolutionStep)

fct_info.SetString("function_name", "ImportData")
CoSimIO.Register(fct_info,           ImportData)

fct_info.SetString("function_name", "ExportData")
CoSimIO.Register(fct_info,           ExportData)

fct_info.SetString("function_name", "ImportMesh")
CoSimIO.Register(fct_info,           ImportMesh)

fct_info.SetString("function_name", "ExportMesh")
CoSimIO.Register(fct_info,           ExportMesh)

# running the simulation
# externally orchestrated
run_info = CoSimIO.Info()
run_info.SetString("connection_name", s_connection_name)
CoSimIO.Run(run_info)

# Disconnecting
disconnect_settings = CoSimIO.Info()
disconnect_settings.SetString("connection_name", s_connection_name)
return_info = CoSimIO.Disconnect(disconnect_settings)
cosimio_check_equal(return_info.GetInt("connection_status"), CoSimIO.ConnectionStatus.Disconnected)