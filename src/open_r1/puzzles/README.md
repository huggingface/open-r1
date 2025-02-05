The puzzles module contains a simple and extensible system for generating and verifying reasoning tasks.
The focus is on tasks where infinite variants can be generated with automatic answer verification, like mathematics, logic puzzles or coding tasks, although
we highly encourage creativity - if you can come up with less STEM-y tasks that can still be rigorously validated, we'd love to see them!

# Generating puzzles

After `pip install`ing the open-r1 repo, you can very quickly get started

```python
>>> from open_r1.puzzles import LinearEquationConfig, LinearEquationTask

>>> task = LinearEquationTask()
>>> # Tasks are iterable, so you can iterate with "for question, answer in task:"
>>> question, answer = next(iter(task))
>>> print(question)
'-2y - 4 = -16'

# To score a model output, use task.validate()
>>> task.verify("y = 6", answer)
1.0

>>> # To control the task difficulty, you can use the task's associated config
>>> config = LinearEquationConfig()
>>> config.min_coefficient = -1000
>>> config.max_coefficient = 1000
>>> harder_task = LinearEquationTask(config)
```

## Adding new puzzles

[add puzzle guide goes here]

## Coming soon:

- Proper indexing of puzzles
- More puzzle types!
- Lazy loading (if the module gets very big)