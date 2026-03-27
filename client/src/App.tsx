import { RouterProvider } from 'react-router-dom';
import { router } from './router/router.ts';

function App() {
    return <RouterProvider router={router} />;
}
export default App;
