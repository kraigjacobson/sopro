class SoproOutputProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = []; this.offset = 0; this.queued = 0; this.started = false; this.ending = false; this.finished = false;
    this.speechGain = 0; this.lastSample = 0; this.armed = false; this.waited = 0;
    this.prime = Math.round(sampleRate * .35); this.primeCap = Math.round(sampleRate * .8);
    this.port.onmessage = ({ data }) => {
      if (data.type === 'chunk') { this.queue.push(data.samples); this.queued += data.samples.length; }
      else if (data.type === 'end') this.ending = true;
      else if (data.type === 'go') this.armed = true;
    };
  }

  next() {
    while (this.queue.length && this.offset >= this.queue[0].length) { this.queue.shift(); this.offset = 0; }
    if (!this.queue.length) return null;
    this.queued--; return this.queue[0][this.offset++];
  }

  remaining() { return this.queued; }

  process(inputs, outputs) {
    const output = outputs[0]?.[0];
    if (!output) return true;
    if (!this.armed) return true;
    if (!this.started && this.queued) {
      if (this.queued < this.prime && !this.ending && this.waited < this.primeCap) {
        this.waited += output.length;
        return true;
      }
    }
    const fadeFrames = Math.max(1, sampleRate * .005), fadeStep = 1 / fadeFrames;
    for (let index = 0; index < output.length; index++) {
      const value = this.next();
      if (value !== null) {
        if (!this.started) { this.started = true; this.port.postMessage({ type: 'start', at: currentTime + index / sampleRate }); }
        this.speechGain = Math.min(1, this.speechGain + fadeStep);
        if (this.ending) this.speechGain = Math.min(this.speechGain, (this.remaining() + 1) / fadeFrames);
        this.lastSample = value;
        output[index] = value * this.speechGain;
      } else if (this.ending) {
        this.speechGain = Math.max(0, this.speechGain - fadeStep); output[index] = this.lastSample * this.speechGain;
        if (!this.speechGain) {
          output.fill(0, index + 1);
          if (!this.finished) { this.finished = true; this.port.postMessage({ type: 'end' }); }
          return false;
        }
      } else {
        this.speechGain = Math.max(0, this.speechGain - fadeStep);
        output[index] = this.lastSample * this.speechGain;
      }
    }
    return true;
  }
}

registerProcessor('sopro-output', SoproOutputProcessor);
