import { io } from "./node_modules/socket.io-client/dist/socket.io.esm.min.js";
import LPF from "./lpf.js"

var socket;

var timeTexts = document.getElementById("timetexts");
var fsTexts = document.getElementById("fstexts");
var counterTexts = document.getElementById("countertexts");

var recordingButton = document.getElementById("recording");
var recordingLabel = document.getElementById("recordingLabel");
let configButton = document.getElementById("setconfig");
let configHeader = document.getElementById("config");
var fileText = document.getElementById("filename");
//var latencyText = document.getElementById("latencytext");
var sensematHolder = document.getElementById("sensemat_holder");
var sensematDiffHolder = document.getElementById("sensemat_diff_holder");

var sync = document.getElementById("sync");
var setpointButton = document.getElementById("setpoint");

var s_0_0_Cart = document.getElementById("s_0_0-head");
var s_0_0_Smoothie;
var s_0_0_Time;
var s_0_0_means = { Mat: s_0_0_Time };

var s_1_1_Cart = document.getElementById("s_1_1-head");
var s_1_1_Smoothie;
var s_1_1_Time;
var s_1_1_means = { Mat: s_1_1_Time };

var meanHeadCart = document.getElementById("mean-head");
var meanHeadSmoothie;
var meanHeadTime;
var means = { Mat: meanHeadTime };

var respNN_Cart = document.getElementById("respNN-head");
var respNN_Smoothie;
var respNN_Time;
var respNN_means = { Mat: respNN_Time };

var board_settings_tpl = document.getElementById("board_settings");

var showValues = false;
// var showDiff = false;
// var sensorCount = 128;
// var rowCount = 16;
var recording = false;
var connected = false;

var sensorWindow = [];
var d_autoscale = {head:1, body:1};
var autoscale = {head:1, body:1};

recordingButton.addEventListener("click", handle_recording);
configButton.addEventListener("click", update_config);
configHeader.addEventListener("click", toggle_expander);
setpointButton.addEventListener("click", handle_setpoint);

class Sensemat {
  constructor(rx, tx, id, d, ledpower, gain, integration, guard, samplerate, ttl) {
    this.rx = rx;
    this.tx = tx;
    this.id = id;
    this.d = d;
    this.ledpower = ledpower;
    this.gain = gain;
    this.integration = integration;
    this.guard = guard;
    this.samplerate = samplerate;
    this.sample_time = 5.2 * 4;
    this.ttl = ttl;
  }

  sensorCount() {
    return this.rx * this.tx;
  }

  fs() {
    let tc = this.tx * (this.integration + this.rx * this.sample_time + this.guard)
    return 1e6 / tc;
  }
}

let configuredSensemats = [
  new Sensemat(
    8,
    16,
    "Mat",
    "6400",
    "4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095",
    1000,
    1000,
    10,
    50,
    1
  ),
];

recordingButton.disabled = true;
recording = false;
let senseMats = {};
let senseMatsDiff = {};
let times = {};
let fss = {};
let counters = {};

function updateSensematDom() {
  senseMats = {};
  senseMatsDiff = {};
  sensematHolder.innerHTML = "";
  sensematDiffHolder.innerHTML = "";
  timeTexts.innerHTML = "";
  fsTexts.innerHTML = "";
  counterTexts.innerHTML = "";
  configuredSensemats.forEach((sensemat, index) => {
    let timeTxt = document.createElement("div");
    times[sensemat.id] = timeTxt;
    timeTexts.append(timeTxt);

    fss[sensemat.id] = 0

    let fsTxt = document.createElement("div");
    fsTxt.innerText = `${sensemat.id}: ${sensemat.samplerate} (max: ${Math.round(sensemat.fs())})`
    fsTexts.append(fsTxt);

    let counterTxt = document.createElement("div");
    counters[sensemat.id] = counterTxt;
    counterTexts.append(counterTxt);

    senseMats[sensemat.id] = [];
    senseMatsDiff[sensemat.id] = [];

    let [sensor_wrapper, sensor_elements] = updateSensematViz(sensemat, index);
    senseMats[sensemat.id] = sensor_elements;
    sensematHolder.appendChild(sensor_wrapper);

    let [sensor_diff_wrapper, sensor_diff_elements] = updateSensematViz(sensemat, index);
    senseMatsDiff[sensemat.id] = sensor_diff_elements;
    sensematDiffHolder.appendChild(sensor_diff_wrapper);
  });
}

function updateSensematViz(sensemat, index) {
  let sensor_wrapper = document.createElement("div");
  sensor_wrapper.className = "sensor_wrapper";
  let sensor_name = document.createElement("h5");
  sensor_name.className = "text-h5";
  sensor_name.innerHTML = `${sensemat.id}`;
  sensor_wrapper.appendChild(sensor_name);

  let sensor_holder = document.createElement("div");
  sensor_holder.className = "sensorholder spacing-mb-5";
  sensor_holder.id = `sensor_${sensemat.id}`;
  sensor_holder.style.gridTemplateColumns = "var(--sensor-size) ".repeat(
    sensemat.rx + 1
  );

  let elements = [];
  for (let column = 0; column < sensemat.rx; column++) {
    let currentColumn = document.createElement("div");
    currentColumn.className = `column`;
    sensor_holder.appendChild(currentColumn);
    for (let row = 0; row < sensemat.tx; row++) {
      let el = document.createElement("span");
      el.id = `sensor_${index}_${row * sensemat.rx + column}`;
      el.className = "sensor";
      el.dataset.setpoint = 0;
      el.dataset.value = 0;
      el.filter = new LPF()
      el.filter.smoothing = 0.5;
      el.filter.init([10,10,10,10,10,10,10,10,10,10]);
      currentColumn.append(el);
      elements.push(el);
    }
  }
  sensor_wrapper.appendChild(sensor_holder);
  return [sensor_wrapper, elements];
}

//updateSensematDom();
render();

document
  .getElementById("showvalues")
  .addEventListener("change", function (event) {
    showValues = event.currentTarget.checked;
  });

connect_gateway();

function handle_setpoint() {
  configuredSensemats.forEach((sensemat, index) => {
    let sensors = senseMatsDiff[sensemat.id];
    for (let s in sensors){
      let sensor = sensors[s];
      sensor.dataset.setpoint = sensor.dataset.value;
    }
  });
}

function handle_recording() {
  if (recording) {
    recordingLabel.innerText = "Start recording";
    recordingButton.classList.remove("signal-error");
    recordingButton.classList.add("signal-success");
    //fileText.style.color = "#000000";
    end_recording();
  } else {
    recordingLabel.innerText = "End recording";
    recordingButton.classList.add("signal-error");
    recordingButton.classList.remove("signal-success");
    // recordingButton.classList.add('rec');
    // fileText.style.color = "#FF0000";
    start_recording();
  }
}

function toggle_expander(event) {
  let target = event.currentTarget;
  target.parentElement.classList.toggle("expanded");
}

function update_config() {
  let configValue = document.getElementById("configValue");
  if (socket?.connected) {
    console.log("Config is: ");
    const config = JSON.parse(configValue.value);
    console.log(config);
    socket.emit("update_config", config);
  }
}

function start_recording() {
  fileText.innerText = "";
  if (socket?.connected) {
    recording = true;
    socket.emit("start_recording");
  }
}

function end_recording() {
  fileText.innerText = "";
  if (socket?.connected) {
    recording = false;
    socket.emit("end_recording");
  }
}

// function supportRespPpgAccSamplesFromOldRecordings(currentSensor, samples) {
//     if (currentSensor == sensorCount) accXTime?.append(+new Date(), samples[sensorCount]); //X
//     if (currentSensor == sensorCount + 1) accYTime?.append(+new Date(), samples[sensorCount + 1]); //Y
//     if (currentSensor == sensorCount + 2) accZTime?.append(+new Date(), samples[sensorCount + 2]); //Z
//     if (currentSensor == sensorCount + 3) ppgTime?.append(+new Date(), samples[sensorCount + 3]); //PPG
//     if (currentSensor == sensorCount + 4) respTime?.append(+new Date(), samples[sensorCount + 4]); //RESP
// }

function extractTimestamp(samples) {
  return samples.shift();
}

function extractCounter(samples) {
  samples.shift();
  return samples.shift();
}

function connect_gateway() {
  if (socket?.connected) {
    console.log("Already connected");
    return false;
  }
  socket = io.connect();

  var last;
  function ping() {
    last = new Date();
    socket.emit("ping_from_client");
  }

  socket.on("connect", function () {
    connected = true;
    recordingButton.disabled = false;
    // if (respCart.getContext) {
    //     render();
    //     window.onresize = render;
    // }
    // ping();
    console.log("Connected");
  });

  socket.on("disconnect", function () {
    connected = false;
    console.log("Disconnected");
  });

  socket.on("server_config_updated", function (config) {
    console.log("Server config updated to:");
    const configString = JSON.stringify(config, null, 2);
    console.log(configString);
    let configValue = document.getElementById("configValue");
    configValue.value = configString;
    //sensorCount = 0;
    // rowCount = 0;
    configuredSensemats = config.map((matConfig) => {
      console.log("Mapping config: ", matConfig.rx, " x ", matConfig.tx);
      //sensorCount = sensorCount + matConfig.width * matConfig.height;
      // rowCount = rowCount + matConfig.height;
      return new Sensemat(
        matConfig.rx,
        matConfig.tx,
        matConfig.id,
        matConfig.d,
        matConfig.ledpower,
        matConfig.gain,
        matConfig.integration,
        matConfig.guard,
        matConfig.samplerate,
        matConfig.ttl
      );
    });
    updateSensematDom();
  });

  socket.on("pong_from_server", function () {
    var latency = new Date() - last;
    latencyText.innerText = latency + "ms";
    setTimeout(ping, 500);
  });

  socket.on("on_recording_started", function (msg) {
    fileText.innerHTML += `recording to ${msg.data} <br/>`;
  });

  socket.on("on_recording_ended", function (msg) {
    fileText.innerHTML += `recorded to <a class="hyperlink" href="/recordings/${msg.data}">${msg.data}</a><br/>`;
  });

  socket.on("on_sample_data", function (packet) {
    if (packet.sensemat) {
      process_sensemat_data(packet.sensemat);
    }
  });
}
function getValueFromSample(sample) {
  return sample.substring(sample.indexOf("=") + 1);
}

function averageValueOf(values) {
  return (
    values.map((v) => parseFloat(v)).reduce((lhs, rhs) => lhs + rhs, 0) /
    values.length
  );
}

function process_sensemat_data(packet) {
  // DATA in format:
  // id, timestamp, sensor_0, sensor_1, ... ,sensor_127, mean, ttl_time, ttl_state
  let diff = 0;
  for (var msg of packet) {
    if (!msg.id) {
      return;
    }

    let id = msg.id;
    let data = msg.data;
    let counter = data.shift();
    let bTime = data.shift();
    let dTime = bTime - fss[id];
    let afs = Math.round(1000000 / dTime);
    fss[id] = bTime;
    counters[id].innerText = id + ": " + counter;
    times[id].innerText = `${id} : ${bTime}`;
    diff = (counter - diff) % 255;
    let senseMatIndex = configuredSensemats.findIndex((item) => item.id == id);
    let senseMat = configuredSensemats[senseMatIndex];
    let sensorCount = senseMat.sensorCount();
    let sensors = senseMats[id];
    let sensorsDiff = senseMatsDiff[id];
    var d = 64000;
    if (senseMat.d) {
      d = senseMat.d;
    }
    var respNN = data.pop();
    respNN_Time.append(+new Date(), respNN);
    data.pop();
    data.pop();
    if (data.length > sensorCount) {

      let mat_avg = data.pop();

      means[id]?.append(+new Date(), mat_avg / 1000);
    }
    s_0_0_means[id]?.append(+new Date(), Math.log10(Math.max(1,data[17])));
    s_1_1_means[id]?.append(+new Date(), Math.log10(Math.max(1,data[68])));

    let d_autoscale_finder = 0;
    let autoscale_finder = 0;
    for (let sensor in data) {
      if (sensor >= sensorCount) {
        //supportRespPpgAccSamplesFromOldRecordings(sensor, data);
        continue;
      } else {
        //const rawval = Math.log10(Math.max(1,data[sensor]));
        const rawval = data[sensor];

        // raw value display
        const val = (rawval / autoscale[id]).toFixed(2);
        const aval = Math.abs(val);
        const displayVal = Math.pow(aval, 1 / 1.4); // Gamma of 1.8

        // diff value opt 1 (using offset)
        // const d_rval = rawval - sensorsDiff[sensor].dataset.setpoint;
        // const d_val = (d_rval / d_autoscale[id]).toFixed(2);
        // const d_aval = Math.abs(d_val);
        // const d_displayVal = Math.pow(d_aval, 1 / 1.4).toFixed(2); // Gamma of 1.8

        //diif value opt 2 using prev_val
        // const d_rval = rawval - sensorsDiff[sensor].dataset.value;
        // const d_val = (d_rval / autoscale).toFixed(2);
        // const d_aval = Math.abs(d_val);
        // const d_displayVal = Math.pow(d_aval, 1 / 1.4).toFixed(2); // Gamma of 1.8

        // Using LPF
        const d_rval = sensorsDiff[sensor].filter.next(rawval -  sensorsDiff[sensor].dataset.setpoint);
        const d_val = (d_rval / d_autoscale[id]).toFixed(2);
        const d_aval = Math.abs(d_val);
        const d_displayVal = Math.pow(d_aval, 1 / 1.4).toFixed(2); // Gamma of 1.8

        d_autoscale_finder = Math.max(d_autoscale_finder, d_rval)
        autoscale_finder = Math.max(autoscale_finder, rawval)
        if (showValues) {
          sensors[sensor].innerText = val;
          sensors[sensor].style.backgroundColor = "rgb(0,0,0,0)";

          sensorsDiff[sensor].innerText = d_val;
          sensorsDiff[sensor].style.backgroundColor = "rgb(0,0,0,0)";
        } else {
          sensors[sensor].innerText = "+";
          sensors[sensor].style.backgroundColor =
            val <= 1
              ? val < 0
                ? `rgb(0,153,87,${displayVal})`
                : `rgb(222,56,53,${displayVal})`
              : `rgb(0,142,255,1)`;

          sensorsDiff[sensor].innerText = "+";
          sensorsDiff[sensor].style.backgroundColor =
          d_rval < 0
                ? `rgb(0,153,87,${d_displayVal})`
                : `rgb(222,56,53,${d_displayVal})`

          // sensors[sensor].style.boxShadow = `inset 0 0 0px ${
          //   5 - 5 * Math.min(aval, 1)
          // }px #ffffff`;
        }
        sensorsDiff[sensor].dataset.value = rawval;
      }
    }
    d_autoscale[id] = d_autoscale_finder;
    autoscale[id] = autoscale_finder*2;
    sensorWindow = data;
  }
  sync.innerText = diff;
}
function disconnect_gateway() {
  if (!socket?.connected) {
    console.log("Nothing to disconnect");
    return false;
  }
  latencySmoothie?.stop();
  respSmoothie?.stop();
  ppgSmoothie?.stop();
  meanSmoothie?.stop();
  socket.disconnect();
}

function setupChart(domElement, color, width, mpp=20) {
  domElement.width = width;
  domElement.height = 50;
  let smoothie = new SmoothieChart({
    millisPerPixel: mpp,
    grid: { fillStyle: "#1e1e1e", borderVisible: false, millisPerLine: 1000 },
    scaleSmoothing: 0.81,
    labels: { disabled: true },
    tooltip: true,
    tooltipLine: { strokeStyle: "#ffffff" },
  });
  smoothie.streamTo(domElement, 10);
  var timeseries = new TimeSeries();
  smoothie.addTimeSeries(timeseries, {
    strokeStyle: color,
    lineWidth: 2,
  });
  return [smoothie, timeseries];
}

function render() {
  [s_0_0_Smoothie, s_0_0_Time] = setupChart(s_0_0_Cart, "#ffffff", 650, 20);
  s_0_0_means = { Mat: s_0_0_Time };
  [s_1_1_Smoothie, s_1_1_Time] = setupChart(s_1_1_Cart, "#ffffff", 650, 20);
  s_1_1_means = { Mat: s_1_1_Time };
  [meanHeadSmoothie, meanHeadTime] = setupChart(meanHeadCart, "#ffffff", 650, 20);
  means = { Mat: meanHeadTime };
  [respNN_Smoothie, respNN_Time] = setupChart(respNN_Cart, "#ffffff", 650, 20);
  respNN_means = { Mat: respNN_Time };
}
