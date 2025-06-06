import KratosMultiphysics

import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
import KratosMultiphysics.KratosUnittest as KratosUnittest
import KratosMultiphysics.kratos_utilities as kratos_utils

from KratosMultiphysics.StructuralMechanicsApplication import structural_mechanics_analysis
from KratosMultiphysics import eigen_solver_factory

from math import sqrt
from cmath import phase

def add_variables(mp):
    mp.AddNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)
    mp.AddNodalSolutionStepVariable(KratosMultiphysics.REACTION)
    mp.AddNodalSolutionStepVariable(KratosMultiphysics.REACTION_MOMENT)
    mp.AddNodalSolutionStepVariable(KratosMultiphysics.ROTATION)
    mp.AddNodalSolutionStepVariable(KratosMultiphysics.VELOCITY)
    mp.AddNodalSolutionStepVariable(KratosMultiphysics.ACCELERATION)
    mp.AddNodalSolutionStepVariable(KratosMultiphysics.VOLUME_ACCELERATION)
    mp.AddNodalSolutionStepVariable(KratosMultiphysics.NODAL_MASS)
    mp.AddNodalSolutionStepVariable(StructuralMechanicsApplication.POINT_LOAD)

def solve_eigen(mp,echo=0):
    from KratosMultiphysics import LinearSolversApplication

    eigensolver_settings = KratosMultiphysics.Parameters("""{
        "solver_type": "feast",
        "symmetric": true,
        "e_min": 1.0e-2,
        "e_max": 25.0,
        "subspace_size": 7
    }""")

    eigen_solver = eigen_solver_factory.ConstructSolver(eigensolver_settings)
    builder_and_solver = KratosMultiphysics.ResidualBasedBlockBuilderAndSolver(eigen_solver)

    eigen_scheme = StructuralMechanicsApplication.EigensolverDynamicScheme()
    mass_matrix_diagonal_value = 1.0
    stiffness_matrix_diagonal_value = -1.0
    eig_strategy = StructuralMechanicsApplication.EigensolverStrategy(mp,
                                                                    eigen_scheme,
                                                                    builder_and_solver,
                                                                    mass_matrix_diagonal_value,
                                                                    stiffness_matrix_diagonal_value)

    eig_strategy.SetEchoLevel(echo)
    eig_strategy.Solve()

def setup_harmonic_solver(mp, echo=0):
    builder_and_solver = KratosMultiphysics.ResidualBasedBlockBuilderAndSolver(KratosMultiphysics.LinearSolver())
    eigen_scheme = StructuralMechanicsApplication.EigensolverDynamicScheme()
    harmonic_strategy = StructuralMechanicsApplication.HarmonicAnalysisStrategy(mp, eigen_scheme, builder_and_solver, False)
    harmonic_strategy.SetEchoLevel(echo)

    return harmonic_strategy

def add_dofs(mp):
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.DISPLACEMENT_X, KratosMultiphysics.REACTION_X,mp)
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.DISPLACEMENT_Y, KratosMultiphysics.REACTION_Y,mp)
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.DISPLACEMENT_Z, KratosMultiphysics.REACTION_Z,mp)
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.ROTATION_X, KratosMultiphysics.REACTION_MOMENT_X,mp)
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.ROTATION_Y, KratosMultiphysics.REACTION_MOMENT_Y,mp)
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.ROTATION_Z, KratosMultiphysics.REACTION_MOMENT_Z,mp)

def create_2dof_geometry(current_model, stiffness, mass, damping=0):
    mp = current_model.CreateModelPart("mdof")
    add_variables(mp)

    base = mp.CreateNewNode(3,0.0,0.0,0.0)
    node1 = mp.CreateNewNode(1,10.0,0.0,0.0)
    node2 = mp.CreateNewNode(2,20.0,0.0,0.0)

    add_dofs(mp)
    for no in mp.Nodes:
        no.Fix(KratosMultiphysics.DISPLACEMENT_Y)
        no.Fix(KratosMultiphysics.DISPLACEMENT_Z)
    base.Fix(KratosMultiphysics.DISPLACEMENT_X)

    #create elements and conditions
    spring1 = mp.CreateNewElement("SpringDamperElement3D", 1, [3,1], mp.GetProperties()[1])
    spring2 = mp.CreateNewElement("SpringDamperElement3D", 2, [2,1], mp.GetProperties()[1])
    mass1 = mp.CreateNewElement("NodalConcentratedElement3D1N", 3, [1], mp.GetProperties()[1])
    mass2 = mp.CreateNewElement("NodalConcentratedElement3D1N", 4, [2], mp.GetProperties()[1])
    mp.CreateNewCondition("PointLoadCondition3D1N",1,[1],mp.GetProperties()[1])
    mp.CreateNewCondition("PointLoadCondition3D1N",2,[2],mp.GetProperties()[1])

    mass1.SetValue(KratosMultiphysics.NODAL_MASS,mass)
    mass2.SetValue(KratosMultiphysics.NODAL_MASS,mass/2)
    spring1.SetValue(StructuralMechanicsApplication.NODAL_DISPLACEMENT_STIFFNESS,[stiffness,0,0])
    spring2.SetValue(StructuralMechanicsApplication.NODAL_DISPLACEMENT_STIFFNESS,[stiffness/2,0,0])
    node1.SetSolutionStepValue(StructuralMechanicsApplication.POINT_LOAD,0,[1,0,0])
    mp.GetProperties()[1].SetValue(StructuralMechanicsApplication.SYSTEM_DAMPING_RATIO, damping)

    return mp

def test_undamped_mdof_harmonic():
    # Analytic solution taken from Humar - Dynamics of Structures p. 675
    current_model = KratosMultiphysics.Model()

    # Material properties
    stiffness = 10.0
    mass = 2.0

    # Create the model
    mp = create_2dof_geometry(current_model, stiffness, mass)

    # Solve the eigenproblem
    solve_eigen(mp)

    # Perform the harmonic analysis
    harmonic_solver = setup_harmonic_solver(mp)
    exfreq = 1.0
    max_exfreq = 1.1
    df = 0.05

    while exfreq <= max_exfreq:
        for condition in mp.Conditions:
            print(f"freq {exfreq} condition {condition.Id} point load {condition[StructuralMechanicsApplication.POINT_LOAD]}")

        mp.CloneTimeStep(exfreq)
        harmonic_solver.Solve()

        omega_ratio = (exfreq / sqrt(stiffness / mass))**2
        disp_x1_expected = ((1.0/3.0 * stiffness) / (0.5 - omega_ratio) + (1.0/1.5 * stiffness) / (2.0 - omega_ratio)) / stiffness**2
        disp_x2_expected = ((2.0/3.0 * stiffness) / (0.5 - omega_ratio) - (2.0/3.0 * stiffness) / (2.0 - omega_ratio)) / stiffness**2

        print(mp.Nodes[1].GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X, 0), disp_x1_expected)
        print(mp.Nodes[2].GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X, 0), disp_x2_expected)
       
        exfreq += df

if __name__ == '__main__':
    test_undamped_mdof_harmonic()
