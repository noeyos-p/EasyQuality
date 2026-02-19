import React, { memo } from 'react';
import { Loader2, CheckCircle2, AlertCircle, Clock } from 'lucide-react';

export interface TaskStatus {
    id: string;
    status: 'waiting' | 'processing' | 'completed' | 'error';
    message: string;
    filename?: string;
    doc_name?: string;
    updated_at: number;
}

interface TaskNotificationProps {
    tasks: TaskStatus[];
    onCloseTask: (id: string) => void;
}

const TaskNotification: React.FC<TaskNotificationProps> = memo(({ tasks, onCloseTask }) => {
    if (tasks.length === 0) return null;

    return (
        <div className="fixed bottom-6 right-6 z-[9990] flex flex-col gap-3 w-80 max-h-[70vh] overflow-y-auto">
            {tasks.map((task) => (
                <div
                    key={task.id}
                    className={`bg-white dark:bg-gray-800 border-l-4 p-4 rounded-lg shadow-xl transform transition-all duration-300 ease-in-out border ${task.status === 'completed' ? 'border-l-green-500' :
                        task.status === 'error' ? 'border-l-red-500' :
                            task.status === 'processing' ? 'border-l-blue-500' : 'border-l-gray-400'
                        }`}
                >
                    <div className="flex justify-between items-start">
                        <div className="flex items-center gap-2">
                            {task.status === 'processing' && <Loader2 className="w-4 h-4 animate-spin text-blue-500" />}
                            {task.status === 'completed' && <CheckCircle2 className="w-4 h-4 text-green-500" />}
                            {task.status === 'error' && <AlertCircle className="w-4 h-4 text-red-500" />}
                            {task.status === 'waiting' && <Clock className="w-4 h-4 text-gray-500" />}
                            <span className="font-semibold text-sm dark:text-gray-100">
                                {task.filename || task.doc_name || '문서 작업'}
                            </span>
                        </div>
                        <button
                            onClick={() => onCloseTask(task.id)}
                            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                    <p className="mt-2 text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                        {task.message}
                    </p>
                    {task.status === 'processing' && (
                        <div className="mt-3 w-full bg-gray-200 dark:bg-gray-700 h-1 rounded-full overflow-hidden">
                            <div className="bg-blue-500 h-full animate-[shimmer_2s_infinite] w-2/3"></div>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
});

export default TaskNotification;
