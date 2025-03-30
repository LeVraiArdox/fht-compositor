# Output configuration

When starting up, `fht-compositor` will scan all connected outputs and turn them on. They will get arranged
in a straight horizontal line, like the following:

![Default output arrangement](/assets/default-output-arrangement.svg)

You refer by outputs using their connector names, for example `eDP-1` is your laptop builtin display,
`DP-{number}` are display port connectors, `HDMI-A-{number}` are builtin HDMI ports, etc.

You configure outputs by using the `outputs.{connector-name}` table.

---

#### `disable`

Whether to completely disable an output. You will not be able to accesss it using the mouse.

> [!NOTE] Disabling an already enabled output
> When you disable a output that has opened windows in its workspaces, these windows will get "merged" or into the same workspaces
> of the *newly active* output instead.

---

#### `mode`

A string representation of a mode that takes the form of `{width}x{height}` or `{width}x{height}@{refresh-hz}`. Optionally, there's
custom mode support using [CVT timing calculation](http://www.uruk.org/~erich/projects/cvt/)

When picking the mode for the output, the compositor will first filter out modes with matching width and height, then
- If there's a given refresh rate, find the mode which refresh rate is the closest to what you have given
- If there's no refresh rate, pick the highest one available.

---

#### `scale`

The *integer scale* of the output. There's currently no support for fractional scaling in `fht-compositor`.

---

#### `position`

The position of the *top-left corner*. The output space is absolute.

> [!NOTE] Overlapping output geometries
> If your configuration contains two overlapping outputs, `fht-compositor` will resort to the default output arragement seen
> at the top of this page. It will also print out a warning message in the logs
