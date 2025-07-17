use std::mem::MaybeUninit;
use std::os::unix::process::CommandExt;
use std::process::Stdio;
use std::time::Duration;

use smithay::reexports::rustix;
use smithay::reexports::wayland_server::backend::Credentials;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{DisplayHandle, Resource};
use smithay::utils::{Coordinate, Point, Rectangle};

#[cfg(feature = "xdg-screencast-portal")]
pub mod pipewire;

pub fn get_monotonic_time() -> Duration {
    // This does the same job as a Clock<Monotonic> provided by smithay.
    // I do not understand why they decided to put on an abstraction
    //
    // We also do not use the Time<Monotonic> structure provided by smithay since its really
    // annoying to work with (addition, difference, etc...)
    let timespec = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    Duration::new(timespec.tv_sec as u64, timespec.tv_nsec as u32)
}

pub fn spawn(cmd: &str) {
    let cmd = cmd.to_string();
    crate::profile_function!();
    let res = std::thread::Builder::new()
        .name("Command spawner".to_string())
        .spawn(move || {
            let mut command = std::process::Command::new("/bin/sh");
            command.args(["-c", &cmd]);
            // Disable all IO.
            command
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());

            // Double for in order to avoid the command being a child of fht-compositor.
            // This will allow us to avoid creating zombie processes.
            //
            // This also lets us not waitpid from the child
            unsafe {
                command.pre_exec(|| {
                    match libc::fork() {
                        -1 => return Err(std::io::Error::last_os_error()),
                        0 => (),
                        _ => libc::_exit(0),
                    }

                    if libc::setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }

                    // Reset signal handlers.
                    let mut signal_set = MaybeUninit::uninit();
                    libc::sigemptyset(signal_set.as_mut_ptr());
                    libc::sigprocmask(
                        libc::SIG_SETMASK,
                        signal_set.as_mut_ptr(),
                        std::ptr::null_mut(),
                    );

                    Ok(())
                });
            }

            let mut child = match command.spawn() {
                Ok(child) => child,
                Err(err) => {
                    warn!(?err, ?cmd, "Error spawning command");
                    return;
                }
            };

            match child.wait() {
                Ok(status) => {
                    if !status.success() {
                        warn!(?status, "Child didn't exit sucessfully")
                    }
                }
                Err(err) => {
                    warn!(?err, "Failed to wait for child")
                }
            }
        });

    if let Err(err) = res {
        warn!(?err, "Failed to create command spawner for command")
    }
}

pub fn get_credentials_for_surface(surface: &WlSurface) -> Option<Credentials> {
    let handle = surface.handle().upgrade()?;
    let dh = DisplayHandle::from(handle);
    let client = dh.get_client(surface.id()).ok()?;
    client.get_credentials(&dh).ok()
}

pub trait RectCenterExt<C: Coordinate, Kind> {
    fn center(self) -> Point<C, Kind>;
}

impl<Kind> RectCenterExt<i32, Kind> for Rectangle<i32, Kind> {
    fn center(self) -> Point<i32, Kind> {
        self.loc + self.size.downscale(2).to_point()
    }
}

impl<Kind> RectCenterExt<f64, Kind> for Rectangle<f64, Kind> {
    fn center(self) -> Point<f64, Kind> {
        self.loc + self.size.downscale(2.0).to_point()
    }
}
