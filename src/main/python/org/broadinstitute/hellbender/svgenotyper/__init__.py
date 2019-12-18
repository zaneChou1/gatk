from .svgenotyper import train, genotype
from .svgenotyper.arguments import parse_args_genotype, parse_args_train
from .svgenotyper.io import write_vcf
from ._version import __version__