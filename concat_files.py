

base_dir = './data/txt/'
filenames = [base_dir+'sketch_labeled.txt', base_dir+'sketch_unl.txt']
with open('sketch_all.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
            