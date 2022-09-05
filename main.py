# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import json


def main():
    with open('post_clean_freq_dist.json') as f:
        data = json.load(f)
        count = 0
        for i in data:
            count += 1
        print(count)
# 90128 words in the dictionary
# they have correct words and incorrect words

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
