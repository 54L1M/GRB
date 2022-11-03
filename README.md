# **GRB**

GRB stands for Gamma-Ray Burst.

* [Enviroment](https://github.com/Perun21/GRB#enviroment)
* [Requirements](https://github.com/Perun21/GRB#requirements)

## **Enviroment**

Install python `virtualenv` package using `pip` : `pip install virtualenv` .

Create a virtual enviroment using :`python -m venv env` .

Install the requirements using : `pip install -r requirements.txt` .

## **Requirements**

For each new package you add to the project, add the name of the package to the `requirements.in` file and use the command below and the `requirements.txt` file will be updated:

```pip-compile requirements.in```
