# ![](icon.png) AutoCoder

<a href="/flairNLP/flair/blob/master/CONTRIBUTING.md"><img src="https://camo.githubusercontent.com/8f697c48adc5026cc6d83dd45e42b9b93ee1803c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f6e747269627574696f6e732d77656c636f6d652d627269676874677265656e2e737667" alt="Contributions welcome" data-canonical-src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" style="max-width:100%;"></a> <a href="https://opensource.org/licenses/apache" rel="nofollow"></a>

#### A basic and simple tool for code auto completion, fine-tuned from the pytorch [pre-trained GPT-2 variants](https://huggingface.co/transformers/pretrained_models.html) offered by the awesome [🤗 transformers](https://github.com/huggingface/transformers) library.

### Demo
![demo](demo.gif)

### Features
- Write with Python or Java.


### Quick Start
Here provides three ways of quick-start. Before that,


#### Load form 🤗transformers models 
Now there are [two fine-tuned models](https://huggingface.co/congcongwang/distilgpt2_fine_tuned_coder) uploded to 🤗transformers models library. They can be used easily as long as you `pip install transformers`


```python
from transformers import AutoTokenizer,AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("congcongwang/gpt2_medium_fine_tuned_coder")
model = AutoModelWithLMHead.from_pretrained("congcongwang/gpt2_medium_fine_tuned_coder")
# or
# tokenizer = AutoTokenizer.from_pretrained("congcongwang/distilgpt2_fine_tuned_coder")
# model = AutoModelWithLMHead.from_pretrained("congcongwang/distilgpt2_fine_tuned_coder")
use_cuda=True
context="def factorial"
lang="python" # can be java as well.

if use_cuda:
    model.to("cuda")

input_ids = tokenizer.encode("<python> " + context,
                                     return_tensors='pt') if lang == "python" else tokenizer.encode(
            "<java> " + context, return_tensors='pt')
outputs = model.generate(input_ids=input_ids.to("cuda") if use_cuda else input_ids,
                         max_length=128,
                         temperature=0.7,
                         num_return_sequences=1)

decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
print(decoded)
```



#### Ready-to-go Interaction
```
git clone <this repository>
pip install -r requirements.txt
```

1. Download the fine-tuned models, here are two versions provided.
    * [distilgpt2_fine_tuned_coder (params: 82M, size: 291MB)](https://ucdcs-student.ucd.ie/~cwang/autocoder/distilgpt2_fine_tuned_coder.zip)
    * [gpt2_medium_fine_tuned_coder.zip (params: 345M, size: 1.22GB)](https://ucdcs-student.ucd.ie/~cwang/autocoder/gpt2_medium_fine_tuned_coder.zip)
2. Unzip the model and move it to `./model` (create it first)
3. Run the interact: `python interact.py`

#### Fine-tuning yours
```
git clone <this repository>
pip install -r requirements.txt
```

1. Preparing [the dataset](./dataset)
2. Start fine-tuning model: `python train.py --model_select distilgpt2` 
3. After fine-tuning, the model will be saved to `./model/distilgpt2_fine_tuned_coder/0_GPTSingleHead` which is exactly the fine-tuned version as provided in Ready-to-go Interaction.

\* For more params setting of training, `python train.py -h`

### Generative examples
Good Python generation examples by fine-tuned GPT2-medium
```python
-------------Example 1--------------------------------------
Context code: def factorial
Generated: 
def factorial(n):
    return 1 if n == 1 else n * factorial(n - 1)

--------------Example 2-------------------------------------
Context code: def sum_of_series(num_of_terms:int, power: int)
Generated:
def sum_of_series(num_of_terms:int, power: int) -> float:
    """Calculate Sum of Series.
    >>> sum_of_series(1, 1, 10)
    55.0
    >>> sum_of_series(1, 10, 100)
    49600.0
    """
    sum = (num_of_terms / 2) * power
    return sum

--------------Example 3-------------------------------------
Context code: def quick_sort(collection)
Generated:
def quick_sort(collection):
    """
    >>> quick_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> quick_sort([])
    []
    >>> quick_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    length = len(collection)
    if length <= 1:
        return collection
    else:
        # Use the last element as the first pivot
        pivot = collection.pop()
        # Put elements greater than pivot in greater list
        # Put elements lesser than pivot in lesser list
        greater, lesser = [], []
        for element in collection:
            if element > pivot:
                ...
```
Good Java generation examples by fine-tuned GPT2-medium

```javascript
--------------Example 1-------------------------------------
Context code: Scanner input= new Scanner(System.in)
Generated:
Scanner input= new Scanner(System.in)
System.out.println("Enter the Octal number: ");
// Take octal number as input from user in a string
String oct = input.next();

--------------Example 2-------------------------------------
Context code: private static int CountCharacters(String str)
Generated:
private static int CountCharacters(String str) {
        return str.replaceAll("\\s", "").length();
}
```
\* Although some generated examples look good, it needs to take a grain of salt to judge the model's actual performance. The model may simply **"remembers"** existing code in the training set well.

### TODO list
- Expand the dataset (and construct the dataset more carefeully) and increase context window. Try larger generative models like GPT-2 large or even [GPT-3 variants](https://arxiv.org/abs/2005.14165) as proposed recently if the computational resources are allowed.
- Remove overlapping between training examples and dev examples for contamination studies. That says, to what extent the model memorizes examples rigidly or [at surface heuristics level during training](https://arxiv.org/pdf/1902.01007.pdf).
- Try some adversarial examples (more complicated for model's reasoning capability testing purpose) to test the robustness of the model.
- Integrate this into real-life use case such as a code editor - [Sublime Text](https://www.sublimetext.com/), where a threshold of joint probability may need to be studied for code snippet recommendations.
- Try some ideas of location-aware code generation. For example, if a human coder is sitting writing a comment, the autocoder should be aware of the coder's context (left and right if available) to help complete the corresponding content.
- Model size and inference efficiency is a problem in real-life use cases.
- Do research in this problem domain to grab a general idea of what work has done in the literature for this particular problem.

### Blog linked to this project
- [The details of dataset construction and fine-tunning process](#) (in plan)

### Extra notes
* For mutli-GPU training, it only works when torch==1.4.0. It will be not working when torch==1.5.0. No idea so far how to fix this issue.