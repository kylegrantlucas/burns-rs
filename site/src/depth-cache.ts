/**
 * Generated depth maps keyed by the shared Rust source key (`depthCacheKey`),
 * which already covers the model hash and preprocessing version. Values are
 * the canonical 16-bit PNG bytes exactly as postprocessed; imported maps are
 * never stored here.
 */
const DATABASE = 'burns-depth';
const STORE = 'maps';

function settled<T>(pending: IDBRequest<T>): Promise<T> {
  const { promise, resolve, reject } = Promise.withResolvers<T>();
  pending.onsuccess = () => resolve(pending.result);
  pending.onerror = () => reject(pending.error ?? new Error('IndexedDB request failed'));
  return promise;
}

async function withStore<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  if (typeof indexedDB === 'undefined') throw new Error('IndexedDB is not available in this browser');
  const opening = indexedDB.open(DATABASE, 1);
  opening.onupgradeneeded = () => opening.result.createObjectStore(STORE);
  const database = await settled(opening);
  try {
    const transaction = database.transaction(STORE, mode);
    const done = Promise.withResolvers<void>();
    transaction.oncomplete = () => done.resolve();
    transaction.onabort = transaction.onerror = () =>
      done.reject(transaction.error ?? new Error('IndexedDB transaction failed'));
    // A failed request also aborts the transaction; its error is reported by the request.
    done.promise.catch(() => undefined);
    const result = await settled(run(transaction.objectStore(STORE)));
    await done.promise;
    return result;
  } finally {
    database.close();
  }
}

export async function cachedDepth(key: string): Promise<Uint8Array | null> {
  const value: unknown = await withStore('readonly', (store) => store.get(key));
  return value instanceof Uint8Array ? value : null;
}

export async function storeDepth(key: string, png: Uint8Array): Promise<void> {
  await withStore('readwrite', (store) => store.put(png, key));
}
