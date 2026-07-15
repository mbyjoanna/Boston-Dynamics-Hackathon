export class BLEDevice {
    constructor() {
        this.device = null;
        this.server = null;
        this.service = null;
        this.writeCharacteristic = null;
        this.notifyCharacteristic = null;
        this.callback = null;
        this.disconnectCallback = null;
    }
    
    async scan() {
        const SERVICE_UUID = '0000fd02-0000-1000-8000-00805f9b34fb';
        this.device = await navigator.bluetooth.requestDevice({
            filters: [{ services: [SERVICE_UUID] }]
        });
    }
    get name() {
        return this.device ? this.device.name : null;
    }
    // FIX: connect() takes UUIDs as args (matching working ble.js)
    // callback is set via .callback property before calling connect()
    async connect(serviceUUID, writeUUID, notifyUUID) {
        try {
            if (!this.device) {
                // fallback: if scan() wasn't called, do a combined scan+connect
                this.device = await navigator.bluetooth.requestDevice({
                    filters: [{ services: [serviceUUID] }]
                });
            }
            this.device.addEventListener(
                'gattserverdisconnected',
                this.handleDisconnect.bind(this)
            );
            this.server  = await this.device.gatt.connect();
            this.service = await this.server.getPrimaryService(serviceUUID);
            this.writeCharacteristic  = await this.service.getCharacteristic(writeUUID);
            this.notifyCharacteristic = await this.service.getCharacteristic(notifyUUID);
            await this.notifyCharacteristic.startNotifications();
            this.notifyCharacteristic.addEventListener(
                'characteristicvaluechanged',
                this.handleNotification.bind(this)
            );
            console.log('BLE connected:', this.device.name);
            return true;
        } catch (error) {
            console.error('BLE connect error:', error);
            return false;
        }
    }
    handleNotification(event) {
        const data = new Uint8Array(event.target.value.buffer);
        if (this.callback) {
            try { this.callback(data); }
            catch (err) { console.error('Notification callback error:', err); }
        }
    }
    handleDisconnect(event) {
        console.log('BLE disconnected:', this.device?.name);
        if (this.disconnectCallback) {
            try { this.disconnectCallback(event); }
            catch (err) { console.error('Disconnect callback error:', err); }
        }
    }
    async send(data) {
        if (!this.writeCharacteristic) throw new Error('Not connected');
        await this.writeCharacteristic.writeValue(new Uint8Array(data));
    }
    disconnect() {
        if (this.device?.gatt?.connected) {
            this.device.gatt.disconnect();
        }
    }
}
