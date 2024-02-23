# Test that datasets are loaded correctly
from src.data import SyntheticPkpdDatasetCollection
from src.data import SyntheticContinuousDatasetCollection

def load_pkpd_dataset():
    synthetic_pkpd_dataset_collection = SyntheticPkpdDatasetCollection(
                                        conf_coeff=10.0,
                                        num_patients=100,
                                        equation_str='EQ_4_D',
                                        seed=100,
                                        window_size=15,
                                        max_seq_length=60,
                                        projection_horizon=5,
                                        lag=0,
                                        cf_seq_mode='sliding_treatment',
                                        treatment_mode='multiclass')
    print('Done')
    
def load_continuous_dataset():
    synthetic_pkpd_dataset_collection = SyntheticContinuousDatasetCollection(
                                        chemo_coeff=10.0,
                                        radio_coeff=10.0,
                                        num_patients={'train': 100, 'val': 10, 'test': 10},
                                        equation_str='EQ_4_D',
                                        seed=100,
                                        window_size=15,
                                        max_seq_length=60,
                                        projection_horizon=5,
                                        lag=0,
                                        cf_seq_mode='sliding_treatment',
                                        treatment_mode='multiclass')
    print('Done')
    
if __name__ == '__main__':
    # load_pkpd_dataset()
    load_continuous_dataset()
    print('Tests passed!')
