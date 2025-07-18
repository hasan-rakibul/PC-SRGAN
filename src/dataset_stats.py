
import os

def get_file_count(directory):
    count = 0
    for param_dir in os.listdir(directory):
            count += len(os.listdir(os.path.join(directory, param_dir)))
    return count

def train_val_test_count(dataset):
    train = os.path.join(dataset, 'train/mesh_7')
    val = os.path.join(dataset, 'validation/mesh_7')
    test = os.path.join(dataset, 'test/mesh_7')

    train_count = get_file_count(train)
    val_count = get_file_count(val)
    test_count = get_file_count(test)
    total = train_count + val_count + test_count
    
    print(dataset)
    print('Train & Validation & Test & Total')
    print(train_count, '&', val_count, '&', test_count, '&', total, '\n')


def main():
    train_val_test_count('data/Allen-Cahn_Periodic')
    train_val_test_count('data/Allen-Cahn_Neumann')
    train_val_test_count('data/Erikson_Johnson')
    train_val_test_count("data/Allen-Cahn_Periodic_x4/")

if __name__ == "__main__":
    main()
