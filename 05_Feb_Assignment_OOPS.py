{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "##Q1. Explain Class and Object with respect to Object-Oriented Programming. Give a suitable example."
      ],
      "metadata": {
        "id": "GoTLYUkxKLPt"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###In Object-Oriented Programming (OOP), a class is a blueprint for creating objects. It defines a set of attributes (data) and methods (behavior) that the objects created from it will have. An object is an instance of a class. It has its own set of attributes and methods and is created by instantiating a class."
      ],
      "metadata": {
        "id": "Zg1HI8hXKgOU"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "fRHummiXKFYc"
      },
      "outputs": [],
      "source": [
        "class Student:\n",
        "    def __init__(self, phone_number, age, degree):\n",
        "        self.phone_number = phone_number\n",
        "        self.age = age\n",
        "        self.degree = degree\n",
        "\n",
        "    def return_students_details(self):\n",
        "        return self.phone_number , self.age , self.degree\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rushi=Student(859125423,21,\"BCA\")"
      ],
      "metadata": {
        "id": "XXzdxXwLNg65"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rushi.degree"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "Mjo6ioVqNyto",
        "outputId": "37599e74-5977-4754-ffad-a9d2c6af3511"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'BCA'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rushi.return_students_details()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tma8ZU4gNkHl",
        "outputId": "7805297d-ab2a-423e-b5d2-b36c357e76ae"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(859125423, 21, 'BCA')"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q2. Name the four pillars of OOPs.\n",
        "\n",
        "####1) Encapsulation: Encapsulation is the practice of hiding the internal workings of an object from the outside world, and exposing only the necessary functionality through a public interface. This helps to ensure that the object's data is not modified or accessed in unintended ways.\n",
        "\n",
        "####2) Abstraction: Abstraction is the process of identifying essential features of an object and ignoring the rest. It allows us to focus on what an object does, rather than how it does it. Abstraction is achieved through the use of abstract classes and interfaces.\n",
        "\n",
        "####3) Inheritance: Inheritance is the process of creating new classes from existing classes. It allows a new class to inherit the properties and methods of an existing class, and to add its own properties and methods. Inheritance helps to reduce code duplication and increase code reusability.\n",
        "\n",
        "####4) Polymorphism: Polymorphism is the ability of an object to take on many forms. It allows us to write code that can work with objects of different classes, as long as they implement the same interface or have the same parent class. Polymorphism is achieved through method overriding and method overloading."
      ],
      "metadata": {
        "id": "wehwdlOjN-y9"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "6qukF5HWNvH7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q3. Explain why the __init__() function is used. Give a suitable example.\n",
        "\n",
        "####The __init__() function is a special method in Python that is used to initialize the instance variables of an object when it is created. It is called a constructor method because it is used to construct or create an instance of a class.\n",
        "####The __init__() method is executed automatically when an object is created, and it is used to set the initial values of the object's attributes. It takes the `self` parameter as the first argument  (we can give any name to it)."
      ],
      "metadata": {
        "id": "aUX3ruRpOruC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class Person:\n",
        "    def __init__(self, name, age):\n",
        "        self.name = name\n",
        "        self.age = age\n",
        "\n",
        "person1 = Person(\"Rushikesh\", 21)"
      ],
      "metadata": {
        "id": "Zm-7AEkaOt04"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "person1.age"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3b7CW4mdPupb",
        "outputId": "f7f574ae-9bec-41e5-fc1f-2eff8bbaa2c3"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "21"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "person1.name"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "-iBr-tcgQDIg",
        "outputId": "958a0f6a-fa1c-4bb7-be34-38fee1cc3922"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Rushikesh'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q4. Why self is used in OOPs?\n",
        "####self is a keyword that is used to refer to the current object or instance of a class. It is used to access the attributes and methods of the object, and to modify its state.\n",
        "###When a method is called on an object, the self keyword is automatically passed as the first argument to the method. This allows the method to access the object's attributes and methods, and to perform operations on its state.\n",
        "\n",
        "####For example, imagine that we have a Person class with an age attribute and a method called get_age() that returns the age of the person. When we create an object of the Person class and call the get_age() method on it, the self keyword refers to the specific instance of the Person object, and allows the method to retrieve the age attribute of that specific instance."
      ],
      "metadata": {
        "id": "CjXKDLLKQpXc"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "fvJS3y_lQkL3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Q5. What is inheritance? Give an example for each type of inheritance.\n",
        "\n",
        "###Inheritance is a key concept in Object-Oriented Programming (OOP) that allows a class to inherit the attributes and methods of another class. Inheritance helps to create a hierarchy of classes and facilitates code reuse and abstraction.\n",
        "\n",
        "###There are four types of inheritance in Python:\n",
        "\n",
        "####1) Single Inheritance: In this type of inheritance, a class inherits the attributes and methods of a single parent class.\n",
        "####Example:"
      ],
      "metadata": {
        "id": "wdRV2eKPReXC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "##Single Inheritance\n",
        "\n",
        "class Animal:\n",
        "    def __init__(self, name):\n",
        "        self.name = name\n",
        "\n",
        "    def eat(self):\n",
        "        print(f\"{self.name} is eating.\")\n",
        "\n",
        "class Dog(Animal):\n",
        "    def __init__(self, name, breed):\n",
        "        super().__init__(name)\n",
        "        self.breed = breed\n",
        "\n",
        "    def bark(self):\n",
        "        print(\"Woof!\")\n",
        "\n",
        "dog1 = Dog(\"Moti\", \"Labrador\")\n",
        "dog1.eat()\n",
        "dog1.bark()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PStjVfvnSMik",
        "outputId": "242e2be6-b334-4b9c-fc27-8fb43d4dc917"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Moti is eating.\n",
            "Woof!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "####2) Multiple Inheritance: In this type of inheritance, a class inherits the attributes and methods of multiple parent classes."
      ],
      "metadata": {
        "id": "BHVaBujaS7P0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "###Multiple Inheritance\n",
        "\n",
        "class Car:\n",
        "    def __init__(self, make, model):\n",
        "        self.make = make\n",
        "        self.model = model\n",
        "\n",
        "    def drive(self):\n",
        "        print(f\"{self.make} {self.model} is driving.\")\n",
        "\n",
        "class Electric:\n",
        "    def __init__(self, range):\n",
        "        self.range = range\n",
        "\n",
        "    def charge(self):\n",
        "        print(\"Charging...\")\n",
        "\n",
        "class Tesla(Car, Electric):\n",
        "    def __init__(self, make, model, range):\n",
        "        Car.__init__(self, make, model)\n",
        "        Electric.__init__(self, range)\n",
        "\n",
        "tesla1 = Tesla(\"Tesla\", \"Model S\", 400)\n",
        "tesla1.drive()\n",
        "tesla1.charge()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "krFTjyN6Sisp",
        "outputId": "3788e625-a793-48e8-d3df-70d91b677ee0"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Tesla Model S is driving.\n",
            "Charging...\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "####3) Multilevel Inheritance: In this type of inheritance, a class inherits from a parent class, which in turn inherits from another parent class.\n",
        "####*Example*:"
      ],
      "metadata": {
        "id": "T2axb4d9TIof"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class Vehicle:\n",
        "    def __init__(self, make, model):\n",
        "        self.make = make\n",
        "        self.model = model\n",
        "\n",
        "    def start(self):\n",
        "        print(\"Starting the engine.\")\n",
        "\n",
        "class Car(Vehicle):\n",
        "    def drive(self):\n",
        "        print(\"Driving the car.\")\n",
        "\n",
        "class ElectricCar(Car):\n",
        "    def charge(self):\n",
        "        print(\"Charging the car.\")\n",
        "\n",
        "electric_car1 = ElectricCar(\"Tesla\", \"Model S\")\n",
        "electric_car1.start()\n",
        "electric_car1.drive()\n",
        "electric_car1.charge()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lrc4HnwjTCB3",
        "outputId": "40131e5a-b9d0-4661-fb88-20ee6cb8068b"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Starting the engine.\n",
            "Driving the car.\n",
            "Charging the car.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "####4) Hierarchical Inheritance: In this type of inheritance, multiple classes inherit from a single parent class.\n",
        "####Example:"
      ],
      "metadata": {
        "id": "k3aCgYnrTbi2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class Vehicle:\n",
        "    def __init__(self, make, model):\n",
        "        self.make = make\n",
        "        self.model = model\n",
        "\n",
        "    def start(self):\n",
        "        print(\"Starting the engine.\")\n",
        "\n",
        "class Car(Vehicle):\n",
        "    def drive(self):\n",
        "        print(\"Driving the car.\")\n",
        "\n",
        "class Truck(Vehicle):\n",
        "    def haul(self):\n",
        "        print(\"Hauling a load.\")\n",
        "\n",
        "car1 = Car(\"Toyota\", \"Corolla\")\n",
        "car1.start()\n",
        "car1.drive()\n",
        "\n",
        "truck1 = Truck(\"Ford\", \"F-150\")\n",
        "truck1.start()\n",
        "truck1.haul()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HxXKixTLTOSW",
        "outputId": "9c44802d-bfba-4709-bf88-ce0699da413c"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Starting the engine.\n",
            "Driving the car.\n",
            "Starting the engine.\n",
            "Hauling a load.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "car1.model"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "zl8uqm9ITwIl",
        "outputId": "1ee68a62-6010-4f61-b8b5-b80d04f7e5bb"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Corolla'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "truck1.model"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "K4a4Oh9mUGj_",
        "outputId": "a970dfac-e590-4268-c604-c861e723613a"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'F-150'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "erXfM12uUL0i"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
