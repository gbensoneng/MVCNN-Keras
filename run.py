from MVCNN.train import train_model
from MVCNN.test import test_model
import os


def ask_train():
    input_valid = False
    while not input_valid:
        prompt_train = input("Do you wish to train a model? (yes/no)").lower()
        if prompt_train == "yes":
            train_model()
            input_valid = True
        elif prompt_train == "no":
            print("Skipping model training.")
            input_valid = True
        else:
            print("Invalid input. Try again.")


def ask_test():
    input_valid = False
    while not input_valid:
        prompt_predict = input("Do you wish to test model on dataset? (yes/no)").lower()
        if prompt_predict == "yes":
            prompt_path = input("Type the full path to the directory containing test data:")
            if not os.path.isdir(prompt_path):
                print("Not a valid directory. Please try again.")
                continue
            prompt_model = input("Specify path to MVCNN model. Otherwise it will be loaded from default location.")
            if not os.path.isfile(prompt_model) and prompt_model != "":
                print("Invalid model path. Please try again.")
                continue
            test_model(prompt_path)
            input_valid = True
        elif prompt_predict == "no":
            print("Skipping model testing.")
            input_valid = True
        else:
            print("Invalid input. Try again.")


def main():
    ask_train()
    ask_test()
    print("Exiting program...")


if __name__ == '__main__':
    main()
