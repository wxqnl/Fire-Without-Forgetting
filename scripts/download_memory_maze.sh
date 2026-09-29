#!/bin/bash
# usage: scripts/download_memory_maze.sh eval train-part0 ... train-part9
# official Memory Maze 9x9 offline dataset (Pasukonis et al. 2022), Google Drive folder 1-tMdUzshBEEIo5EjBnECn7IFK1lJLFsQ
# A plain download of a popular file gets Drive's "Quota exceeded" page, a bounded byte range does not:
# fetch 512 MB ranges and append them, so an interrupted download resumes at the last full chunk.
DEST="$(dirname "$0")/../data/memory_maze_9x9_official"; mkdir -p "$DEST"; cd "$DEST"
declare -A IDS=(
  [eval]=18YPeLnu_TkVx7r4mZjeQ9gSMPEI-jtiP
  [train-part0]=1EdWafDjZG3VUo7CXTERx7fTx4IqWWNvC [train-part1]=1Kyn8seqBiU8drJ8uyS2RSgeefLatUCzi
  [train-part2]=1FaxOJJu6hbPjkab5dtCu-QWxMVDEcj5T [train-part3]=1S6G-knZUG2V0Jhobseb_2-_V5n2jhVMS
  [train-part4]=1br0-EZfh4aTY5E66_m-PXR9i0v1IA1-8 [train-part5]=17ZGVQ8fGVv9FlEQiLy-j_M9r4NlABY3i
  [train-part6]=10vPLCDjv4AC35TkdwAkgUrrAvAhIQJ9v [train-part7]=1vebIvddC4UG78YSpbGTJlr-Hl1ScZeGV
  [train-part8]=1KmVoAofGWnwBJ0EqClYqWNBzENMA8riE [train-part9]=1N4eiw0DV-HrxWSkhRDmBV6skGTiOUz5B)
CHUNK=$((512 * 1024 * 1024))
for name in "$@"; do
  url="https://drive.usercontent.google.com/download?id=${IDS[$name]}&export=download&confirm=t"
  [ -f "$name.zip" ] && unzip -tq "$name.zip" >/dev/null 2>&1 && { echo "$(date '+%F %T') $name already ok"; continue; }
  total=$(curl -sL --max-time 60 -r 0-0 -D - -o /dev/null "$url" | grep -i '^content-range' | sed 's/.*\///' | tr -d '\r')
  [ -z "$total" ] && { echo "$(date '+%F %T') $name no size"; continue; }
  fails=0
  while :; do
    have=$(stat -c %s "$name.zip" 2>/dev/null || echo 0)
    [ "$have" -ge "$total" ] && break
    end=$((have + CHUNK - 1)); [ $end -ge $total ] && end=$((total - 1))
    code=$(curl -sL --max-time 1800 -r $have-$end -o "$name.chunk" -w '%{http_code}' "$url")
    if [ "$code" = 206 ] && [ "$(stat -c %s "$name.chunk")" -eq $((end - have + 1)) ]; then
      cat "$name.chunk" >> "$name.zip"; fails=0
    else
      fails=$((fails + 1)); [ $fails -ge 10 ] && { echo "$(date '+%F %T') $name stuck at $have (http $code)"; break; }
      sleep 30
    fi
  done
  rm -f "$name.chunk"
  if unzip -tq "$name.zip" >/dev/null 2>&1; then echo "$(date '+%F %T') $name ok $(du -h $name.zip | cut -f1)"
  elif [ "$(stat -c %s "$name.zip" 2>/dev/null || echo 0)" -ge "$total" ]; then echo "$(date '+%F %T') $name corrupt, removed"; rm -f "$name.zip"
  fi
done
