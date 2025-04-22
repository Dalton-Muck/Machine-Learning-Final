# CS 4830 Machine-Learning-Final

## Project Overview

These models aim to predict the presence of breast cancer cells in cancer patients. We aim to achieve the highest possible prediction accuracy by using preprocessing techniques (Feature Selection, Dimensionality Reduction, Min/Max Scaling, etc.). After visualization of the raw dataset we decided to implement the support vector machine model and the perceptron model because of their linear separability capabilities. 

## Members and Tasks

- **Colin**: Feature Selection pre-dimensionality-reduction

- **Carson**: SVM Training

- **Brock**: Dimensionality-Reduction

- **Dalton**: Perceptron Training

## Demo Links

- [Draw.io](https://drive.google.com/file/d/1ihIpkdWM_BPVOFXVoz63fD0dn3xw08mE/view?usp=sharing)

- [Powerpoint](https://catmailohio-my.sharepoint.com/:p:/r/personal/bk893421_ohio_edu/Documents/MachineLearningFinalPresentation.pptx?d=wfe7699d32c504f6895ed7ffb3d7d93f8&csf=1&web=1&e=H32d5M)

## Setup

- Download and install [UV](https://github.com/astral-sh/uv)
- Run `uv sync`
- Enter virual environment with `source .venv/bin/activate`
- Add dependencies to `pyproject.toml` under the `dependencies` section

## Results

Using the uniform manifold approximation and projection dimensionalized data to train a Support Vector Machine with a linear kernel we were able to correctly predict the presence of breast cancer cells "96.92%" of the time!
