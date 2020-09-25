[Hoppity](https://openreview.net/pdf?id=SJeqs6EFvB) is a learning based approach to detect and fix bugs in Javascript programs. 

Hoppity is trained on a dataset of (buggy, fixed) pairs from Github commits. Refer to our [gh-crawler](https://github.com/AI-nstein/gh-crawler) repo for scripts to download and generate a dataset. Alternatively, we provide a cooked dataset of pairs with a single AST difference here: https://drive.google.com/file/d/1kEJBCH1weMioTcnmG6fmqz6VP-9KjH7x/view?usp=sharing (you can also find [the json format of cooked code graphs](https://drive.google.com/file/d/17DYDo3g9U0Z8W8bOFw0zOmCjDmkJkP0c/view?usp=sharing))

We also provide trained models for each of the following datasets: <br />
[One Diff Model](https://drive.google.com/file/d/1uULZtgvGz-k_ILMlZW2jPk6QdrQ2q1VN/view?usp=sharing) <br />

[Zero and One Diff Model](https://drive.google.com/file/d/1xAnJwPEd1DzsxHW2Z_SLZikgiUwS6_zW/view?usp=sharing) <br />

[Zero, One, and Two Diff Model](https://drive.google.com/file/d/1z9slfwb2YqC8T71zhWjWFGbNir10A7LA/view?usp=sharing)


# INSTALL

- Install python packages:

```
pip install torch==1.3.1
pip install numpy
pip install -r requirements.txt
```

- Other dependencies

```
cd deps/torchext
pip install -e .
```

- install current package

```
hoppity$ pip install -e .
```

- JS packages

```
npm install shift-parser
npm install ts-morph
npm install shift-spec-consumer
```

# Data Preprocessing

If you would like to use your own dataset, you will need to "cook" the folder as a preprocessing step to generate graphs. 
You can use the `data_process/run_build_main.sh` script to create the cooked dataset. Set the variables `data_root`, `data_name` and `ast_fmt` accordingly. 

This builds the AST in our graph format, for each file and saves it in a pkl file. Additionally, it creates a graph edit file text file for each pair of (buggy, fixed) JS ASTs. This is in a JSON format such that each edit is an object in a list.

# Data Split - Train, Validate, and Test

If you're using the cooked dataset we provided, this portion is already done for you. Once you've downloaded the compressed file, unzip by running `tar xzf cooked-one-diff.gz`. If you do not specify an output directory, the files will be placed in `~/cooked-full-fmt-shift_node/` by default. This will take around an hour. After the files are extracted you can move onto the next step to begin training.

Otherwise, run `data_process/run_split.sh` to partition your cooked dataset. The raw Javascript source files are needed for this script to filter out duplicates. Set the `raw_src` variable in the script accordingly. 

`run_split.sh` calls `split_train_test.py` to load triples from the `save_dir` and partition according to the percentage arguments specified in `config.py`. The default split is 80% train, 10% validation, and 10% test. It saves three files: `test.txt`, `val.txt`, and `train.txt` in the `save_dir` with the cooked data. Each sample name in the cooked dataset is written in one of the three files.

# Training

Now, run `run_main.sh` to train on our pre-processed dataset. Set the variables in the script accordingly. Hyperparameters can be changed in `common/config.py.` The training runs indefinitely. Kill the script manually to end training. 


# Finding the Best Model 

To find the "Best Model", we've provided a script that evaluates each epoch's model dump on the validation set. Run `find_best_model.sh` to start the evaluation. Set the variables accordingly. The loss of each epoch's model will be recorded in the `LOSS_FILE.` 

# Evaluation

We provide an evaluation script that can evaluate a particular model on a number of metrics: 

* Total End-to-End Accuracy - A sample is considered accurate if the model detects the bug and predicts the entire given fix. 
* Location Accuracy - Bug detection acccuracy
* Operator Accuracy - Since there are only 4 operators (ADD, REMOVE, REPLACE_VAL, REPLACE_TYPE), we always report top-1 accuracy. 
* Value Accuracy - If the sample is a REPLACE_VAL or ADD, it is considered accurate if the value is predicted correctly. We also include an UNKNOWN value for literals not included in the vocabulary. If the model predicts UNKNOWN a vlaue not in the vocabulary, it is considered correct. 
* Type Accuracy - If the sample is a REPLACE_TYPE or ADD, it is considered accurate if the node type is predicted correctly.

We also include an option for accuracy breakdown per operation type. Lastly, if you would like an exhaustive evaluation of all metrics, we provide the `output_all` option.
