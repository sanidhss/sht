from txgnn import TxData

d = TxData(data_folder_path='/work/hdd/bisa/sshah21/primekg')
d.prepare_split(split='complex_disease', seed=42)
print("done")
