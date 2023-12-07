import argparse


def function():
    print("Function called!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for Training Model to extract simulated 3D RF")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    args = parser.parse_args()