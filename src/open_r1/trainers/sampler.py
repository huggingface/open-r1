from torch.utils.data import RandomSampler
from typing import Iterator

class RepeatBatchRandomSampler(RandomSampler):
    def __init__(self, 
                 *args,
                num_generations: int = 1, 
                batch_size: int = 3,  
                 **kwargs,
                 )-> None:
        self.num_generations = num_generations
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)
    
    def __len__(self) -> int:
        return super().__len__() * self.num_generations

    def __iter__(self) -> Iterator[int]:
        batch_indices = []
        for idx in super().__iter__():
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                batch_indices = batch_indices * self.num_generations
                yield from batch_indices
                batch_indices = []
                
                
                
if __name__ == "__main__":
    sampler = RepeatBatchRandomSampler(num_generations=2, data_source=range(12), replacement=False)
    # print(list(sampler))
    
    for sample in sampler:
        print(sample)