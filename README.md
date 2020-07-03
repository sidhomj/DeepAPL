# DeepAPL

## Deep learning for distinguishing morphological features of Acute Promyelocytic Leukemia (APL)

Acute Promyelocytic Leukemia (APL) is a subtype of Acute Myeloid Leukemia (AML), classified by a translocation between chromosomes 15 and 17 [t(15;17)], that is notably distinguished clinically by a rapidly progressive and fatal course. Due to the acute nature of its presentation, rapid and accurate diagnosis is required to initiate appropriate therapy that can be curative. However, the gold standard genetic tests can take days to confirm a diagnosis and thus therapy is often initiated on high clinical suspicion based on both clinical presentation as well as direct visualization of the peripheral smear. While there are described cellular morphological features that distinguish APL, there is still considerable difficulty in diagnosing APL from direct visualization of a peripheral smear by a hematopathologist. We hypothesized that deep learning pattern recognition would have greater discriminatory power and consistency compared to humans to distinguish t(15;17) translocation positive APL from t(15;17) translocation negative AML. By applying both cell-level and patient-level classification, we demonstrate that deep learning is capable of distinguishing APL in both a discovery and prospective independent cohort of patients. Furthermore, we extract learned information from the trained network to identify previously undescribed morphological features of APL. Importantly, the deep learning method we describe herein allows a quick,  explainable, and accurate physician aid for diagnosing APL at time of presentation in any medical setting capable of generating a peripheral smear.

In this github repository, all data and code are present to replicate the results illustrated in the manuscript. Data can be found under the Data directory. All code required to recreate the main figure in the manuscript can be found under scripts organized by SC (Single-Cell) and WF (Whole-File) directories.

## Publication
For full description of analysis and approach, refer to the following manuscript:

***

## Dependencies
See requirements.txt for all dependencies to run the analysis.

## Questions, Commments?
For questions or help, email: jsidhom1@jhmi.edu