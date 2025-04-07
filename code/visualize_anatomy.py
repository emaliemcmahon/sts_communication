import nibabel as nib
from nilearn.plotting import plot_anat
import argparse

def plot_img(file, out=None):
    if not out:
        out = 'anat.jpg'
    img = nib.load(file)
    plot_anat(img, output_file=out)


def main():
    parser = argparse.ArgumentParser(description='Plot anatomy')
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to the anatomical image')
    parser.add_argument('--image', '-i', type=str, help='Output image file')
    args = parser.parse_args()
    plot_img(args.file, args.image)

if __name__ == '__main__':
    main()