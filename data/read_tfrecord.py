import tensorflow as tf
import json
import os
import tqdm


def transform_tfrecord_to_json(tfrecord_paths, json_path, num_examples=1, overwrite=False):
    if not overwrite:
        assert os.path.exists(json_path) == False, "json file already exists"
    # Remove if exists
    if os.path.exists(json_path):
        os.remove(json_path)

    count_example = 0
    examples = {}
    # Iterate over all tfrecord files
    for tfrecord_path in tqdm.tqdm(tfrecord_paths):
        # Read TFRecord file
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

        # Iterate over all examples
        for raw_record in raw_dataset.take(num_examples):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            screen_info = {}
            for key, feature in example.features.feature.items():
                kind = feature.WhichOneof('kind')
                # Decode bytes to string if necessary
                screen_info[key] = [
                    v.decode('utf-8') if isinstance(v, bytes) else v 
                    for v in getattr(feature, kind).value
                    ]
            examples[f'example_{count_example}'] = screen_info
            count_example += 1

        
    print(f'Writing {len(examples)} examples to {json_path}')
    # Write and append if exists to json file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f)
    

# create main
if __name__ == '__main__':
    # Path to TFRecord file for test 10% of data
    tfrecord_test_paths = ['/app/data/rico_sca/tfexample/one_shot_rico_subtoken_0.tfrecord']
    print(f'Test TFRecord file paths: {tfrecord_test_paths}\n')

    # Paths to TFRecord files for dev 20% of data
    tfrecord_dev_paths = [f'/app/data/rico_sca/tfexample/one_shot_rico_subtoken_{i}.tfrecord'
                          for i in range(1, 3)]
    print(f'Validation TFRecord file paths: {tfrecord_dev_paths}\n')
    
    # Paths to TFRecord files for training 70% of data
    tfrecord_train_paths = [f'/app/data/rico_sca/tfexample/one_shot_rico_subtoken_{i}.tfrecord'
                            for i in range(3, 10)]
    print(f'Train TFRecord file paths: {tfrecord_train_paths}\n')

    # Creation of samples
    json_path = '/app/data/sample_data/rico_sca_5_examples.json'  # Move to tests/sample_data/rico_sca_5_examples.json
    transform_tfrecord_to_json(tfrecord_test_paths, json_path, num_examples=5, overwrite=True)

    # Sample for train, dev, and test
    json_train_sample_path = '/app/data/rico_sca/rico_sca_train_sample_sample.json'
    json_dev_sample_path = '/app/data/rico_sca/rico_sca_dev_sample_sample.json'
    json_test_sample_path = '/app/data/rico_sca/rico_sca_test_sample_sample.json'
    transform_tfrecord_to_json([tfrecord_train_paths[0]], json_train_sample_path, num_examples=1, overwrite=True)
    transform_tfrecord_to_json([tfrecord_dev_paths[0]], json_dev_sample_path, num_examples=1, overwrite=True)
    transform_tfrecord_to_json([tfrecord_test_paths[0]], json_test_sample_path, num_examples=1, overwrite=True)

    # All samples
    json_train_path = '/app/data/rico_sca/rico_sca_train.json'
    json_dev_path = '/app/data/rico_sca/rico_sca_dev.json'
    json_test_path = '/app/data/rico_sca/rico_sca_test.json'
    transform_tfrecord_to_json(tfrecord_train_paths, json_train_path, num_examples=-1, overwrite=True)
    transform_tfrecord_to_json(tfrecord_dev_paths, json_dev_path, num_examples=-1, overwrite=True)
    transform_tfrecord_to_json(tfrecord_test_paths, json_test_path, num_examples=-1, overwrite=True)




