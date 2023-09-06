import matplotlib.pyplot as plt

x = range(1,15)

mpnn_y = [50.46069,70.96952,83.18364,79.35596,89.14568,84.28035,96.34012,82.95220,102.86598,104.85938,97.88687,96.79718,100.98809,98.24675]

plt.plot(x, avg_y, color="g", marker="o")
plt.plot(x, avg_window_y, color="b", marker="v")
plt.plot(x, lstm_y, color="y", marker="^")

plt.plot(x, mpnn_y, color="r", marker=".")

# Add title and axis names
plt.title('France (27/12/2020-27/03/2021)')
plt.xlabel('Prediction for dt days ahead')
plt.ylabel('AVG error in no of cases')

plt.legend()
plt.savefig("../images/prediction_days.pdf", bbox_inches ="tight")
plt.show()
