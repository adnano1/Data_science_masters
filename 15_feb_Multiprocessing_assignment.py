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
        "###Q1. What is multiprocessing in python? Why is it useful?\n",
        "\n",
        "* multiprocessing allows you to take advantage of the processing power of multiple CPU cores to perform tasks faster. By using multiprocessing, you can run multiple tasks simultaneously, each on a separate CPU core, which can significantly reduce the time it takes to complete a computation.\n",
        "\n",
        "* Multiprocessing is particularly useful for CPU-bound tasks, such as mathematical computations or data analysis. It can also be used for I/O-bound tasks, such as reading or writing large files, where multiple processes can help to improve performance by overlapping I/O operations.\n",
        "\n",
        "####Overall, multiprocessing can help you write more efficient and scalable Python code, which can be especially useful for large-scale data processing, scientific computing, and other computationally intensive applications."
      ],
      "metadata": {
        "id": "kmUaH_FVLuwp"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q2. What are the differences between multiprocessing and multithreading?\n",
        "\n",
        "####Multiprocessing and multithreading are both techniques used to achieve parallelism in a program, but they differ in several ways:\n",
        "\n",
        "* Definition: Multiprocessing involves running multiple processes in parallel on multiple CPU cores, while multithreading involves running multiple threads within a single process.\n",
        "\n",
        "* Memory: Each process has its own memory space, whereas threads share memory within a process.\n",
        "\n",
        "* Isolation: Processes are isolated from each other and can only communicate through inter-process communication (IPC) mechanisms like pipes or sockets, while threads can share data within the same process.\n",
        "\n",
        "* Overhead: Creating a new process incurs a higher overhead than creating a new thread, due to the need to allocate separate memory space and system resources for each process.\n",
        "\n",
        "* Resource utilization: Processes can take advantage of multiple CPU cores and distribute the workload among them, while threads can only run on a single CPU core.\n",
        "\n",
        "* Fault tolerance: If a process crashes, it does not affect other processes, while a thread crash can cause the entire process to crash.\n",
        "\n",
        "####Overall, multiprocessing is generally better suited for CPU-bound tasks that can be parallelized, while multithreading is better suited for I/O-bound tasks that can benefit from concurrent execution. However, the choice between multiprocessing and multithreading ultimately depends on the specific requirements and constraints of the program."
      ],
      "metadata": {
        "id": "M_gkZGF9MJb1"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RJb-S_ssLnCJ"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q3. Write a python code to create a process using the multiprocessing module."
      ],
      "metadata": {
        "id": "n80X6ES7Mvs9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import multiprocessing\n",
        "\n",
        "def test1():\n",
        "    print(\"This is my process.\")\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    my_process = multiprocessing.Process(target=test1)\n",
        "    my_process.start()\n",
        "    my_process.join()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hhBCyvdNMwx8",
        "outputId": "f0133590-a55c-48d3-cabe-95a1bc94e626"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "This is my process.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q4. What is a multiprocessing pool in python? Why is it used?\n",
        "\n",
        "* A multiprocessing pool in Python is a way of creating a group of worker processes that can be used to execute tasks in parallel. The pool creates a fixed number of worker processes and distributes tasks to them, with each worker process executing one task at a time.\n",
        "\n",
        "* The multiprocessing module in Python provides the Pool class for creating a process pool. You can create a Pool object with a specified number of worker processes, and then use the map or apply method to distribute tasks to the workers.\n",
        "\n",
        "* The map method takes an iterable of tasks and applies a function to each task in parallel, returning the results in the same order as the original iterable. The apply method applies a function to a single task in the pool and returns the result.\n",
        "\n",
        "* Multiprocessing pools are used to parallelize the execution of CPU-bound tasks, such as heavy computations or data processing. They can significantly improve the performance of such tasks by distributing the workload across multiple CPU cores, which can result in faster execution times.\n",
        "\n"
      ],
      "metadata": {
        "id": "Mpu8wiSyM9ZI"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q5. How can we create a pool of worker processes in python using the multiprocessing module?"
      ],
      "metadata": {
        "id": "fO58YF3UNsAP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import multiprocessing\n",
        "\n",
        "def test2(x):\n",
        "    return x**2\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    with multiprocessing.Pool(processes=4) as pool:\n",
        "        results = pool.map(test2, [1, 2, 3, 4, 5])\n",
        "    print(results)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SdSB71KOM1J0",
        "outputId": "fa2e5ebc-52fa-41ed-fc96-607fba9809a7"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1, 4, 9, 16, 25]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q6. Write a python program to create 4 processes, each process should print a different number using the multiprocessing module in python."
      ],
      "metadata": {
        "id": "0sISo9WSN3f5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import multiprocessing\n",
        "\n",
        "def print_number(num):\n",
        "    print(f\"Process {multiprocessing.current_process().name} prints {num}\")\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    processes = []\n",
        "    for i in range(4):\n",
        "        p = multiprocessing.Process(target=print_number, args=(i,))\n",
        "        processes.append(p)\n",
        "        p.start()\n",
        "    for p in processes:\n",
        "        p.join()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Lntipo3YNcWi",
        "outputId": "5890916d-6408-40bc-d571-eab6406bb14b"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Process Process-7 prints 0\n",
            "Process Process-8 prints 1\n",
            "Process Process-10 prints 3\n",
            "Process Process-9 prints 2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "5IJdVeibOGrc"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
