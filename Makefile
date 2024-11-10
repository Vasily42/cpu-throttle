.PHONY: install uninstall clean distclean

TARGET ?= target/release/cpu-throttle
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
DESTDIR ?= 

all: $(TARGET)

$(TARGET): 
	cargo build --release

install: $(TARGET)
	mkdir -p  $(DESTDIR)$(BINDIR)
	install -m755 -o root -g root $(TARGET) $(DESTDIR)$(BINDIR)/
	mkdir -p $(DESTDIR)/usr/lib/systemd/system
	install -m644 -o root -g root misc/*.service $(DESTDIR)/usr/lib/systemd/system/
	mkdir -p $(DESTDIR)/etc/cpu-throttle/profiles/
	install -m644 -o root -g root misc/profiles/* $(DESTDIR)/etc/cpu-throttle/profiles/

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/cpu-throttle
	rm -rf $(DESTDIR)/usr/lib/systemd/system/cpu-throttle*
	rm -rf $(DESTDIR)/etc/cpu-throttle

clean:
	rm -f $(TARGET)

distclean:
	cargo clean
