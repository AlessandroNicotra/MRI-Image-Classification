import matplotlib.pyplot as plt

def plot_bar(data):
    fig, ax = plt.subplots(layout='constrained')
    bars = plt.bar(data.keys(), data.values())
    ax.bar_label(bars, label_type='center')
    bars[-1].set_color('red')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.show()

def plot_graph(train, val, title):
    plt.figure(figsize=(10, 5), layout='constrained')
    plt.title(title)
    plt.plot(train, label='train accuracy', color='blue')
    plt.plot(val, label='val accuracy', color='orange')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

def plot_graph_save(train, val, title, directory, type):
    plt.figure(figsize=(10, 5), layout='constrained')
    plt.title(title)
    plt.plot(train, label='train accuracy', color='blue')
    plt.plot(val, label='val accuracy', color='orange')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(directory + '/model_{}.png'.format(type))
    plt.show()

def plot_save(data, directory):
    fig, ax = plt.subplots(layout='constrained')
    bars = plt.bar(data.keys(), data.values())
    ax.bar_label(bars, label_type='center')
    bars[-1].set_color('red')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.savefig(directory + '/model.png')
    plt.show()