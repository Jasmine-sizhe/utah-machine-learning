# utah-machine-learning
This is a machine learning library developed by Sizhe(Jasmine) Chen for CS6350 in University of Utah

Instructions for running code:
1. For Decision Tree:
excute the code with run.sh, then you will see the prompt and then you could choose the dataset and max_depth to get the train test error for each tree.

2. For ensemble learning:
excute the code with run.sh, (if not working please just try 'python main.py', then the code will excecute)
This time you also need to select which dataset you want to process, then it will automatically run adaboost, bagging, and random forest.
Due to computation limiation, I hardcode the iteration from (1,11) for training in a reasonable time. The figure will be displayed after the excecution.

3. For Linear Regression:
excute the code with run.sh, (if not working please just try 'python main.py', then the code will excecute)
This time you need to select which optimization method you would like to try, then it will automatically run with that method and display figure.
In addition, you could also refer to linear_output.ipynb for the output figure under 'Linear Regression' folder if do not have enough time to run.

4. For Perception:
excute the code with run.sh, (if not working please just try 'python main.py', then the code will excecute)
This time you need to select which percetron algorithm you would like to try. After exeuction, it will prompt you input 'Algorithm? 0 for Standard, 1 for Voted, 2 for Average', then just input 0 or 1 or 2 for algorithm specification. Then it will display each epoch of weights updates and train test errors. In addition, you could also refer to output.ipynb for the output figure under 'Perceptron' folder if do not have enough time to run.

5. For SVM:
excute the code with run.sh, (if not working please just try 'python main.py', then the code will excecute)
This time you need to select which SVM algorithm you would like to try. After exeuction, it will prompt you input 'Algorithm? 0 for Primal SVM, 1 for Dual SVM, 2 for Gaussian Kernel', then just input 0 or 1 or 2 for algorithm specification. Then it will display each epoch of weights updates and train test errors. 

6. For Neural Network.:
excute the code with run.sh, (if not working please just try 'python main.py', then the code will excecute)
This time you do need to input any instructions. The code will automatically Train 50 epochs for random initialization, Train 50 epochs for zero initialization, Pytorch Implementation. You could also refer to output.ipynb under Neural Networks folder.