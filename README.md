# Deep learning for distinguishing morphological features of Acute Promyelocytic Leukemia

## Abstract
Acute Promyelocytic Leukemia (APL) is a subtype of Acute Myeloid Leukemia (AML), classified by a translocation between chromosomes 15 and 17 [t(15;17)], that is considered a true oncologic emergency though appropriate therapy is considered curative. Therapy is often initiated on clinical suspicion, informed by both clinical presentation as well as direct visualization of the peripheral smear. We hypothesized that genomic imprinting of morphologic features learned by deep learning pattern recognition would have greater discriminatory power and consistency compared to humans, thereby facilitating identification of t(15;17) positive APL. By applying both cell-level and patient-level classification linked to t(15;17) PML/RARA ground-truth, we demonstrate that deep learning is capable of distinguishing APL in both a discovery and prospective independent cohort of patients. Furthermore, we extract learned information from the trained network to identify previously undescribed morphological features of APL. The deep learning method we describe herein potentially allows a rapid, explainable, and accurate physician-aid for diagnosing APL at the time of presentation in any resource-poor or -rich medical setting given the universally available peripheral smear.

## Data & Code
In this github repository, all data and code are present to replicate the results illustrated in the manuscript. Data can be found under the Data directory. All code required to recreate the main figure in the manuscript can be found under scripts organized by SC (Single-Cell) and WF (Whole-File) directories. Finally all deep learning models are present in python packaged named DeepAPL found in the main directory.

## Publication
For full description of analysis and approach, refer to the following manuscript:

***

## Dependencies
See requirements.txt for all dependencies to run the analysis.

## Questions, Commments?
For questions or help, email: jsidhom1@jhmi.edu