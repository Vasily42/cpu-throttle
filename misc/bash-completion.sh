#!/bin/bash

_cpu_throttle() {
    local cur prev words cword commands config_dir config_files

    _init_completion || return
    
    commands="pause continue toggle read-config exit at switch-config help"
    
    config_dir="/etc/cpu-throttle/profiles/"

    case $prev in
        switch-config)
            if [[ -d $config_dir ]]; then
                config_files=$(ls "$config_dir"*.json 2>/dev/null | xargs -n 1 basename | sed 's/\.json$//')
                COMPREPLY=( $(compgen -W "$config_files" -- ${cur}) )
            fi
            return
            ;;
        cpu-throttle)
            COMPREPLY=( $(compgen -W "$commands" -- ${cur}) )
            return
            ;;
    esac
}

complete -F _cpu_throttle cpu-throttle
