### Instructions to run the project

#### If models are already trained and saved

Ensure models are saved in the paths specified at the beginning of `model_tester.py`

1. Set the path to a CAPTCHA image at the beginning of `model_tester.py`
2. Run the `model_tester.py` to solve the CAPTCHA.

#### If models are not trained yet

Dataset A's captchas should be placed in `samples/type_A` directory
Dataset B's captchas should be placed in `samples/type_B` directory (unless specified differently at the beginning of the dataset_builder scripts)

1. Run `dataset_A_builder.py` and `dataset_B_builder.py` to generate the datasets.
2. Run all the blocks in `Model_A.ipynb` and `Model_B.ipynb` notebooks to train and save the models.
3. Ready for prediction! You can now run the `model_tester.py` demo to solve a CAPTCHA.
