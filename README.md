# 3D-biomembrane-vesicle-shapes-Lagrange-multiplier-approach-
Neural network solver for the 3D shape of vesicles made of lipid bilayers (membrane) calculated by a phase-field approach to the Helfrich model

This is an NN solver for the 3D shape of vesicles that are made of lipid bilayers, which are modeled by a continuum model i.e., phase field. The governing PDE for the shape of the vesicle is the well-known Helfrich elastic bending energy subject to constraints on the surface area and the volume. We have also introduced another contstraint on the center of mass, which ensures that the vesicle remains anchored to the origin of the simulation box. The initial data for the shape is provided as trained parameters for a spherical vesicle in the saved_model_0 folder. 

The user needs to assign the required constraint on the volume or the surface area ideally using sys.argv[1]. This code is written for a single implementation of applying a constraint where a 20000-epoch training is recommended.

In the Lagrange multiplier approach, the constraints on the surface area, volume and center of mass are imposed by including these constraints by appropriate Lagrange multipliers in the cost(loss) function used for training (optimizing the parameters of the NN). The Lagrange multipliers need to be initialized to values that can be found in the literature for a specific shape. We recommend Seifert et. al. Phys. Rev. A 44, 1182 – Published 1 July, 1991. The method of updating for the Lagrange multiplier is the steepest ascent. For more information on this NN solver check the paper by author of this code: Soft Matter, 2024,20, 5359-5366. 
