import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# plot your ECG

# Turn on the minor ticks on
ax.minorticks_on()

# Make the major grid
ax.grid(which='major', linestyle='-', color='red', linewidth='1.0')
# Make the minor grid
ax.grid(which='minor', linestyle=':', color='black', linewidth='0.5')
plt.show()