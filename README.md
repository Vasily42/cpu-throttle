# cpu-throttle

A simple daemon to dynamically limit CPU frequency to prevent the CPU from staying at extremely high temperature for too long, while maintaining the highest performance as possible

## Installation

Only linux supported

```sh
git clone https://github.com/Vasily42/cpu-throttle
cd cpu-throttle
cargo build --release
```

Note: only tags are considered usable, so you may need to checkout latest tag.

Then link or copy binary (to /usr/local/bin/) or use cargo --install

Also you can link/copy .service to /etc/systemd/system/ and enable unit:
```sh
systemctl enable cpu-throttle@85
```

## Usage

If the daemon is already running in the background using systemctl, it is possible for any user to control its behavior in this way: 
```sh
cpu-throttle at 70
# to change target temperature

cpu-throttle pause
# to pause throttling (daemon will wait for next command without exiting)

cpu-throttle continue
# to continue throttling

cpu-throttle help
# to view other commands
```

Keep in mind that the program will not limit the frequency for a few seconds after the CPU starts to load heavily to prevent it from reacting to relatively harmless temperature spikes. 

Also, to prevent sudden loss of performance in case of throtting, the program reduces the frequency gradually until the required temperature is reached

On multi-core systems, the frequency will be limited only on cores that cause long term load (to preserve interactivity)

It is also important to understand that the program does not recognize high load on gpu, so frequency limitation in this case may be excessive, but the solution to this can only be a temporary shutdown of the daemon (e.g. with cpu-throttle pause).





