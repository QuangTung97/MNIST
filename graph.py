import matplotlib.pyplot as plt


def read_stat():
    global train_a, train_c, test_a, test_c
    fp = open('stat.txt', 'r')
    train_a = map(float, fp.readline().split())
    train_c = map(float, fp.readline().split())
    test_a = map(float, fp.readline().split())
    test_c = map(float, fp.readline().split())
    fp.close()


def draw_graph():
    global train_a, train_c, test_a, test_c
    size = len(train_a)
    plt.figure(1)
    plt.subplot(211)
    plt.xlabel("nsamples")
    plt.ylabel("cross entropy")
    plt.plot(range(1, size + 1), train_c, label='train', color='r')
    plt.plot(range(1, size + 1), test_c, label='test', color='b')
    plt.legend()

    plt.subplot(212)
    plt.xlabel("nsamples")
    plt.ylabel("accuracy")
    plt.plot(range(1, size + 1), train_a, label='train', color='r')
    plt.plot(range(1, size + 1), test_a, label='test', color='b')
    plt.legend()
    plt.show()


read_stat()
draw_graph()
