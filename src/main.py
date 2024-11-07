import csv
import os
import pandas as pd
from predict import predict_urls
from train import train_models



def main():
    while True:
        print("\nSelect an option (1-4):")
        print("1. Predict a URL")
        print("2. Train the model")
        print("3. Exit")

        option = input("Enter your choice: ")
        if option == '1':
            url = input("Enter the URL to predict: ")
            # Ensure the URL is in a list format
            prediction = predict_urls([url])
            print("\nPrediction Result:")
            for url, classification in prediction.items():
                print(f"{url}: {classification}")
        elif option == '2':
            os.system('train.py')
        elif option == '3':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()


            

