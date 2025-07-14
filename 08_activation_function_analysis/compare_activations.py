import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#sample-dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

activations = ['logistic', 'tanh', 'relu']
histories = {}


for act in activations:
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation=act,
                        solver='adam', learning_rate_init=0.01,
                        max_iter=200, random_state=1, verbose=False)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    histories[act] = {
        'loss_curve': clf.loss_curve_,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

plt.figure(figsize=(10, 6))
for act in activations:
    plt.plot(histories[act]['loss_curve'], label=f"{act.capitalize()}")
plt.title("Loss Curve Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Final Accuracy Scores:")
for act in activations:
    print(f"{act.capitalize()} - Train: {histories[act]['train_acc']:.2f}, Test: {histories[act]['test_acc']:.2f}")
