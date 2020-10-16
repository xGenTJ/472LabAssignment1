Github: https://github.com/xGenTJ/472LabAssignment1
-----------------------------------------------------------------------------------------------------------------
                                          PROJECT INFO
-----------------------------------------------------------------------------------------------------------------
COMP 472 -  Artificial Intelligence
Lab Assignment: 1
October 19th, 2020
Students:  Tao Jing:	40041626						[Team Leader]
           Priscilla Cournoyer: 27710690
		       Naimur Rashid: 40027867
--------------------------------
                                          PROJECT SETUP
-----------------------------------------------------------------------------------------------------------------
Python 3.0
scikit-learn
-----------------------------------------------------------------------------------------------------------------
                                           INSTRUCTIONS
---------------------------------------------------------------------------------------------------------------  
To run the classification models for the english alphabet, respectively comment & uncomment lines 242-247 in the Main.py file.
To run the classification models for the greek alphabet, respectively comment & uncomment lines 257-262 in the Main.py file.
Output files are in the output folder.


-----------------------------------------------------------------------------------------------------------------
                                  RESPONSIBILITES AND CONTRIBUTIONS
---------------------------------------------------------------------------------------------------------------  
EVERYONE: 
	readCSV()
	plotAlphabet()
	getReplacedLastColumn()
	cleanUpData()
	exportToCSV()
	main
	instancePredictedClass()
	getReversedDic()
	
TAO:
	classifyPerceptron()
    GaussianNaiveBayes()

PRISCILLA:
	calculateConfusionMatrix()
	calculateClassificationReport()
	baselineDecisionTree()
	betterPerformingDecisionTree()

NAIMUR:
	baseMLP
	bestMLP
