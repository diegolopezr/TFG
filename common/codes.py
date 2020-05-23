import argparse as ap

# algorithms files
from common import crc, hamming


def main():
    parser = ap.ArgumentParser(
        description='Encoder/Decoder for network communication - CRC/Hamming Code')

    parser.add_argument('-a', '--algorithm', nargs=1, metavar='algorithm', type=str,
                        choices=['crc', 'hamming'],
                        help='Name of the algorithm to be used. Must be "crc" or "hamming"')
    parser.add_argument('-e', '--encode', nargs=1, metavar='bits_list', type=list, dest='encode',
                        help='Encodes a list of bits and returns the encoded chain')
    parser.add_argument('-d', '--decode', nargs=1, metavar='bits_list', type=list, dest='decode',
                        help='Decodes a list of bits and returns the decoded chain')
    parser.add_argument('-k', '--key', nargs=1, metavar='key_polinom', type=list, dest='key',
                        help='Generator polinom')

    args = parser.parse_args()

    # choose algorithm
    if args.algorithm != None and (args.encode != None or args.decode != None):
        algorithm = args.algorithm[0]
        result = 'NO RESULT'

        if args.encode:
            if algorithm == 'crc':
                result = crc.encode(args.encode[0], args.key[0])
            elif algorithm == 'hamming':
                result = hamming.encode(args.encode[0])
        elif args.decode:
            if algorithm == 'crc':
                result = crc.decode(args.decode[0], args.key[0])
            elif algorithm == 'hamming':
                result = hamming.decode(args.decode[0])

        print(result)


if __name__ == "__main__":
    main()