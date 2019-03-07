Basic API

Generic interfaces:
    Ord
    Index
    Individual
    Records
    Join

Individual operators:
    Estimator
    Recorder
    
Evolutionary operators:
    MateSelector
    Crossover
    Mutator
    SelectionPolicy


The GenericEvolver higher-order operator implements the following pipeline:
    1. Start initial records if none were provided (calls AgeChildRecorder)
    2. Select mates (calls GenericSelector) for ...
    3. Update records
    4. Crossover mates (calls ) ...
    5. Mutate children ...
    6. Start new records
    7. Join parent and child populations
    8. Apply selection policy

`evolve` repeats the process for `generations`, calling callbacks after each 
generation and feeding `individuals`, `records` and `operators` ...


