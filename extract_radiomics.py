import os
import re
import six
import glob
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

from radiomics import featureextractor

def extract(impath, gtpath):
    '''
    Extract radiomics features from a set of images
    '''
    wd = os.path.realpath(os.path.dirname(__file__))
    params = os.path.join(wd, 'Params.yaml')
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    extractor.enableAllFeatures()

    images = sorted(list(glob.iglob(impath, recursive=True)))
    masks = sorted(list(glob.iglob(gtpath, recursive=True)))

    dirpath = os.path.commonpath(masks)

    labels = np.unique(nib.load(masks[0]).get_fdata()).astype(np.int16)
    labels = labels[labels>0]

    assert len(images) == len(masks), \
        'Found different number of images versus masks: {} vs. {}'.format(len(images), len(masks))

    # Set columns names
    print('Found images and masks such as', images[0], masks[0])
    result = extractor.execute(images[0], masks[0], label=int(labels[0]))
    aux = []
    for key, _ in six.iteritems(result):
        if 'original_' != key[:9]:
            continue
        aux.append(key.lstrip('original'))
    cols = []
    for lb in labels:
        cols.extend(['lb{}'.format(int(lb)) + s for s in aux])

    # Extract radiomics features for all found images
    colsn = ['id'] + cols
    df = pd.DataFrame(data=np.zeros((len(images), len(colsn))), columns=colsn)
    for i, image in enumerate(images):
        df.loc[i, 'id'] = image
        print('Extracting radiomics for:')
        print(' - image: ', image)
        print(' - mask: ', masks[i])
        for lb in labels:
            extractor = featureextractor.RadiomicsFeatureExtractor(params)
            try:
                result = extractor.execute(image, masks[i], label=int(lb))
                for key, val in six.iteritems(result):
                    if 'original_' != key[:9]:
                        continue
                    df.loc[i, re.sub(r'original', 'lb{}'.format(int(lb)), key)] = val
            except ValueError as err:
                print(' extraction failed for this label: {}. Error:'.format(lb))
                print(err)
                continue
    df.to_csv(os.path.join(dirpath, 'radiomics_features.csv'), index=False)

    return True


def get_args():
    parser = argparse.ArgumentParser(description='Extract radiomics features from given data set',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--imglob', type=str, default='',
                        help='Path and pattern for images files.', dest='imglob')
    parser.add_argument('-g', '--gtglob', type=str, default='',
                        help='Path and pattern for groundtruth files.', dest='gtglob')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Cardiac images can be found by runing 
    # python extract_radiomics.py -i "path/**/brainmask.nii.gz" -g "path/**/aparc.a2009s+aseg.nii.gz"
    extract(args.imglob, args.gtglob)
