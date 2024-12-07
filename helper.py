
import matplotlib.pyplot as plt
from IPython import display

plt.ion()
figure, axis = plt.subplots(figsize=(5, 3))

def plot_progress(scores, mean_scores):
    axis.clear()
    axis.plot(scores, label="Score", color='blue')
    axis.plot(mean_scores, label="Mean Score", color='green')
    axis.set_title('Training Progress')
    axis.set_xlabel('Number of Games')
    axis.set_ylabel('Score')
    axis.text(len(scores) - 1, scores[-1], str(scores[-1]), color='blue', fontsize=10)
    axis.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]), color='green', fontsize=10)
    axis.set_ylim(bottom=0)
    axis.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    display.clear_output(wait=True)
    display.display(figure)
    plt.pause(0.1)
