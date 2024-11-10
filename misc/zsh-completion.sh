#compdef cpu-throttle

_cpu_throttle() {
  local -a commands
  commands=(
    "pause:Pause throttling"
    "continue:Continue throttling"
    "toggle:Pause/Continue throttling"
    "read-config:Read config.json again and apply it"
    "exit:Exit all threads, finish with success code"
    "at:Set target temperature"
    "switch-config:Switch config.json to a profile and apply it"
    "help:Print help"
  )

  if [[ $words[2] == switch-config ]]; then 
         _values 'configs' $(ls /etc/cpu-throttle/profiles/*.json 2>/dev/null | xargs -n 1 basename | sed 's/\.json$//')
         return 
  fi

  if (( CURRENT < 3 )); then
    _describe 'commands' commands
  fi
}

compdef _cpu_throttle cpu-throttle