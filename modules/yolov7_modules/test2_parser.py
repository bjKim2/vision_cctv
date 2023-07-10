import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test1', type = int, help='merong')
    parser.add_argument('--save-json', action='store_false', help='save a cocoapi-compatible JSON results file')
    opt = parser.parse_args()
    print(opt.test1 + 1)
    print(opt.save_json)
    print(opt)
    batch = 32
    n = 4
    assert batch % n == 0 , 'batch merong'