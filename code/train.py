from dataset import create_generators

train_gen, val_gen, test_gen = create_generators("dataset_ready")

print(train_gen.class_indices)
