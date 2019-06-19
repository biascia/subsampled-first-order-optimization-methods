import json
import matplotlib.pyplot as plt

sgd = json.load(open('sgd.json'))
svrg = json.load(open('svrg.json'))
trish = json.load(open('trish.json'))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(sgd['training']['accuracy'])
ax1.plot(svrg['training']['accuracy'])
ax1.plot(trish['training']['accuracy'])
ax1.title.set_text('Training accuracy')
ax1.legend(['SGD', 'SVRG', 'TRish'])

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(sgd['test']['accuracy'])
ax2.plot(svrg['test']['accuracy'])
ax2.plot(trish['test']['accuracy'])
ax2.title.set_text('Testing accuracy')

plt.show()