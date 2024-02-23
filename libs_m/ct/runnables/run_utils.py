from hydra.utils import instantiate
import shelve

def get_dataset(args):
    ct_db_record = str(args.dataset)
    if args.force_recache:
        dataset_collection = instantiate(args.dataset, _recursive_=True)
        with shelve.open("ct_datasets") as db:
            db[ct_db_record] = dataset_collection
    elif args.load_from_cache:
        try:
            with shelve.open("ct_datasets") as db:
                dataset_collection = db[ct_db_record]
        except KeyError:
            dataset_collection = instantiate(args.dataset, _recursive_=True)
            with shelve.open("ct_datasets") as db:
                db[ct_db_record] = dataset_collection
    else:
        dataset_collection = instantiate(args.dataset, _recursive_=True)
    return dataset_collection